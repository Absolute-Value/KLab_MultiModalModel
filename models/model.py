import os

import numpy as np
import torch
from torch import Tensor, nn
from transformers import ResNetModel, Swinv2Model, T5Config, T5EncoderModel, T5ForConditionalGeneration, logging

# from models.transformer import Transformer, TransformerConfig
from models.vqgan import VQModel
from modules.losses import FocalLoss

logging.set_verbosity_error()

def make_compute_loss_fn(ignore_index,reduction="sum"):
    def compute_crossentopy_loss(logits: Tensor, tgt_ids: Tensor,weight: Tensor | None = None) -> list[Tensor,Tensor]:
        # logits = logits[..., :-1, :].reshape(-1, logits.shape[-1])
        # tgt_ids = tgt_ids[..., 1:].reshape(-1).long()
        
        if reduction == "sum":
            if weight is not None:
                weight = weight.to(tgt_ids.device)
                sample_mask = (tgt_ids.detach().clone() != ignore_index).to(tgt_ids.device)
                sample_vocab_index = tgt_ids[sample_mask]
                sample_size = weight[sample_vocab_index].sum()
                
            else:
                sample_size = (tgt_ids.detach().clone() != ignore_index).sum().to(tgt_ids.device)
        elif reduction == "mean":
            sample_size = None
        else:
            raise NotImplementedError
        
        sum_loss = torch.nn.functional.cross_entropy(logits, tgt_ids, ignore_index=ignore_index, reduction=reduction, weight=weight)
        return sum_loss, sample_size
    
    return compute_crossentopy_loss

# モデルの定義
class MyModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.result_dir = args.result_dir
        vocab_size = 32128 + args.loc_vocab_size + args.additional_vocab_size
        
        if args.vae_ckpt_path != "":
            self.vae = VQModel(ckpt_path=args.vae_ckpt_path).requires_grad_(False)
            self.vae.eval()
        else:
            self.vae = None

        self.language_model = T5EncoderModel.from_pretrained(args.language_model_name).requires_grad_(False)  # device_map="auto"
        if args.language_model_train:
            language_model_config = T5Config.from_pretrained(args.language_model_name)
            self.language_model.encoder.embed_tokens = nn.Embedding(vocab_size, language_model_config.d_model).requires_grad_(True)

        if "resnet" in args.image_model_name:  # 事前学習用に書き換えたのでおそらく動かない
            self.image_model = ResNetModel.from_pretrained(args.image_model_name).requires_grad_(args.image_model_train)
        elif "swinv2" in args.image_model_name:
            # self.image_model = Swinv2Model.from_pretrained(args.image_model_name, use_mask_token=args.pretrain).requires_grad_(args.image_model_train)
            self.image_model = Swinv2Model.from_pretrained(args.image_model_name).requires_grad_(args.image_model_train)
            # self.num_patches = (self.image_model.config.image_size // self.image_model.config.patch_size) ** 2
            self.num_patches = 16**2

        if args.stage == 'classify':
            if False: # Original Transformer
                transformer_config = TransformerConfig(vocab_size=vocab_size, d_model=args.transformer_d_model, num_heads=args.transformer_num_heads, d_ff=args.transformer_d_ff, num_layers=args.transformer_num_layers, max_length=args.max_target_length)
                self.transformer = Transformer(transformer_config)
            else:
                transformer_config = T5Config(
                    vocab_size=vocab_size,
                    d_model=args.transformer_d_model,
                    d_ff=args.transformer_d_ff,
                    d_kv=args.transformer_d_kv,
                    num_heads=args.transformer_num_heads,
                    num_layers=args.transformer_num_layers,
                    max_length=args.max_target_length,
                )
                self.transformer = T5EncoderModel(transformer_config).requires_grad_(True)
                self.transformer.shared.requires_grad_(False)
        else:
            transformer_config = T5Config(
                vocab_size=vocab_size,
                d_model=args.transformer_d_model,
                d_ff=args.transformer_d_ff,
                d_kv=args.transformer_d_kv,
                num_heads=args.transformer_num_heads,
                num_layers=args.transformer_num_layers,
                num_decoder_layers=args.transformer_num_decoder_layers,
                decoder_start_token_id=0,
                max_length=args.max_target_length,
            )
            self.transformer = T5ForConditionalGeneration(transformer_config).requires_grad_(True)

        if args.ffn:
            self.language_ffn = nn.Linear(self.language_model.config.d_model, self.transformer.config.d_model)
            self.image_ffn = nn.Linear(self.image_model.num_features, self.transformer.config.d_model)

        if args.stage == 'classify':
            self.emb_cls_token = EmbClassToken(self.transformer.config.d_model)
            if 'imagenet' in args.datasets:
                self.classifier = nn.Linear(self.transformer.config.d_model, 1000)
            elif 'sun397' in args.datasets:
                self.classifier = nn.Linear(self.transformer.config.d_model, 397)
            elif 'places365' in args.datasets:
                self.classifier = nn.Linear(self.transformer.config.d_model, 365)
            elif 'openimage' in args.datasets:
                self.classifier = nn.Linear(self.transformer.config.d_model, 599)
            elif 'mscoco' in args.datasets:
                self.classifier = nn.Linear(self.transformer.config.d_model, 92)
            else:
                raise NotImplementedError
        
        if args.stage == 'classify':
            ignore_index = -100
        else:
            ignore_index = 0
        if args.loss == 'CrossEntropy':
            if args.stage in ['pretrain','classify']:
                self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
            else:
                self.criterion = make_compute_loss_fn(ignore_index,reduction="sum")
        elif args.loss == 'FocalLoss':
            self.criterion = FocalLoss(ignore_index=ignore_index)
        else:
            raise NotImplementedError

    def forward(self, images, src_texts, src_attention_masks=None, tgt_texts=None, tgt_attention_masks=None, return_loss=True, return_score=False, num_beams=1, num_return_sequences=1, do_sample=False, early_stopping=False):
        if self.args.float_type == 'bfloat16':
            dtype = torch.bfloat16 
        elif self.args.float_type == 'float16':
            dtype = torch.float16
        else:
            dtype = torch.float32

        if src_attention_masks is None:
            src_attention_masks = torch.ones_like(src_texts, device=self.language_model.device, dtype=torch.bool)
            src_attention_masks[src_texts == 0] = 0

        with torch.autocast(device_type='cuda', dtype=dtype, enabled=True if self.args.float_type == 'bfloat16' else False):
            language_embeddings = self.language_model(src_texts, attention_mask=src_attention_masks).last_hidden_state

        # if image_mask_ratio > 0:  # 画像パッチにマスクをかける
        #     bool_masked_pos = self.random_patch_masking(len(images), image_mask_ratio)
        # else:
        #     bool_masked_pos = None
        # image_embeddings = self.image_model(pixel_values=images, bool_masked_pos=bool_masked_pos).last_hidden_state
        with torch.autocast(device_type='cuda', dtype=dtype, enabled=True):
            image_embeddings = self.image_model(pixel_values=images).last_hidden_state

        if self.args.ffn:
            with torch.autocast(device_type='cuda', dtype=dtype, enabled=True):
                language_embeddings = self.language_ffn(language_embeddings)
                image_embeddings = self.image_ffn(image_embeddings)

        concated_embeddings = torch.cat((image_embeddings,language_embeddings), dim=1)
        if self.args.stage == 'classify':
            concated_embeddings = self.emb_cls_token(concated_embeddings)

        image_attention_mask = torch.ones(image_embeddings.shape[0], image_embeddings.shape[1], device=self.image_model.device)
        if self.args.stage == 'classify':
            cls_attention_mask = torch.ones(image_embeddings.shape[0], 1, device=image_embeddings.device)
            concat_attention_mask = torch.cat((cls_attention_mask, image_attention_mask, src_attention_masks), dim=1)
        else:
            concat_attention_mask = torch.cat((image_attention_mask, src_attention_masks), dim=1)

        if return_loss:
            if self.args.stage == 'classify':
                with torch.autocast(device_type='cuda', dtype=dtype, enabled=True):
                    outputs = self.transformer(inputs_embeds=concated_embeddings, attention_mask=concat_attention_mask)
                    sequence_output = outputs[0]
                    logits = self.classifier(sequence_output[:, 0, :])
                    loss = self.criterion(logits, tgt_texts)
                preds = torch.argmax(logits, dim=1)
                return loss, preds
            else:
                if tgt_attention_masks is None:
                    tgt_attention_masks = torch.ones_like(tgt_texts, device=tgt_texts.device, dtype=torch.bool)
                    tgt_attention_masks[tgt_texts == 0] = 0
                with torch.autocast(device_type='cuda', dtype=dtype, enabled=True):
                    logits = self.transformer(inputs_embeds=concated_embeddings, labels=tgt_texts, attention_mask=concat_attention_mask, decoder_attention_mask=tgt_attention_masks).logits
                    if self.args.stage == 'pretrain':
                        loss = self.criterion(logits.view(-1,logits.shape[2]), tgt_texts.view(-1))
                    else:
                        loss, sample_size = self.criterion(logits.view(-1,logits.shape[2]), tgt_texts.view(-1))
                preds = torch.argmax(logits, dim=2)
                if self.args.stage == 'pretrain':
                    return loss, preds
                else:
                    return loss, preds, sample_size
        else:
            if self.args.stage == 'classify':
                with torch.autocast(device_type='cuda', dtype=dtype, enabled=True):
                    outputs = self.transformer(inputs_embeds=concated_embeddings, attention_mask=concat_attention_mask)
                    sequence_output = outputs[0]
                    generated = self.classifier(sequence_output[:, 0, :])
            else:
                with torch.autocast(device_type='cuda', dtype=dtype, enabled=True):
                    if return_score:
                        generated = self.transformer.generate(
                            inputs_embeds=concated_embeddings,
                            attention_mask=concat_attention_mask,
                            num_beams=num_beams,
                            num_return_sequences=num_return_sequences,
                            do_sample=do_sample,
                            early_stopping=early_stopping,
                            max_length=self.args.max_target_length,
                            return_dict_in_generate=True, 
                            output_scores=True
                        )
                        scores = self.transformer.compute_transition_scores(
                            generated.sequences, generated.scores, generated.beam_indices
                        )
                        return generated, scores
                    else:
                        generated = self.transformer.generate(
                            inputs_embeds=concated_embeddings,
                            attention_mask=concat_attention_mask,
                            num_beams=num_beams,
                            num_return_sequences=num_return_sequences,
                            do_sample=do_sample,
                            early_stopping=early_stopping,
                            max_length=self.args.max_target_length,
                        )
            return generated

    def random_patch_masking(self, batch_size, image_mask_ratio):
        len_keep = int(self.num_patches * image_mask_ratio)
        noise = torch.rand(batch_size, self.num_patches, device=self.image_model.device)

        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        mask = torch.ones([batch_size, self.num_patches], device=self.image_model.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return mask

    def image_to_z(self, images):
        z = self.vae.get_codebook_indices(images)  # VAEで中間表現を得る
        z_text = z.cpu().numpy().astype(str)  # 文字列に変換
        z_text = np.char.add(np.char.add('<img_', z_text), '>')  # <extra_id_0>のようにする
        z_text = [''.join(b) for b in z_text]
        return z_text, z

    def z_to_image(self, z):
        x = self.vae.decode_code(z)
        return x

    def save(self, result_name="best.pth"):
        result_path = os.path.join(self.args.result_dir, result_name)
        checkpoints = {'transformer': self.transformer.state_dict()}
        if self.args.image_model_train:
            checkpoints['image_model'] = self.image_model.state_dict()
        if self.args.ffn:
            checkpoints['language_ffn'] = self.language_ffn.state_dict()
            checkpoints['image_ffn'] = self.image_ffn.state_dict()
        if self.args.stage == 'classify':
            checkpoints['emb_cls_token'] = self.emb_cls_token.state_dict()
            checkpoints['classifier'] = self.classifier.state_dict()
        torch.save(checkpoints, result_path)

    def load(self, result_name="best.pth", result_path=None):
        if result_path is None:
            result_path = os.path.join(self.args.result_dir, result_name)
        else:
            result_path = os.path.join(result_path, result_name)
        checkpoints = torch.load(result_path)
        self.transformer.load_state_dict(checkpoints['transformer'])
        if self.args.image_model_train:
            self.image_model.load_state_dict(checkpoints['image_model'])
        if self.args.ffn:
            self.language_ffn.load_state_dict(checkpoints['language_ffn'])
            self.image_ffn.load_state_dict(checkpoints['image_ffn'])
        if self.args.stage == 'classify':
            self.emb_cls_token.load_state_dict(checkpoints['emb_cls_token'])
            self.classifier.load_state_dict(checkpoints['classifier'])

    def load_from_original(self, model_name="google/flan-t5-base"):
        original_model = T5ForConditionalGeneration.from_pretrained(model_name)
        tmp_state_dict = original_model.decoder.state_dict()
        tmp_state_dict.pop('embed_tokens.weight', None)
        self.transformer.decoder.load_state_dict(tmp_state_dict, strict=False)

class EmbClassToken(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
    
    def forward(self, x):
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        return torch.cat((cls_tokens, x), dim=1)
