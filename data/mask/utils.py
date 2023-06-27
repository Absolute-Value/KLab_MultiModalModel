import re
import random

def _merge_consecutive_elements(lst):
    return [element for i, element in enumerate(lst) if i == 0 or element != lst[i-1]]

def _replace_mask(string):
    # <mask>をマッチさせる正規表現パターン
    pattern = re.compile(r"<mask>")

    # 置換カウンターの初期化
    counter = 1

    def replace(match):
        nonlocal counter
        # 置換文字列に番号を付けて作成
        replacement = "<mask_{}>".format(counter)
        counter += 1
        return replacement
    result = re.sub(pattern, replace, string)
    return result

def make_mask_textpair(src_str:str,mask_rasio:float=0.15):
    src_str_list = src_str.split(" ")
    index_list = list(range(len(src_str_list)))
    choice_num = round(len(index_list)*mask_rasio + 0.6)
    choice_num = round(len(index_list)*mask_rasio + 0.6)
    mask_list = random.sample(index_list,choice_num)
    mask_list.sort()

    src_list = _merge_consecutive_elements([src_str_list[index] if not index in mask_list else "<mask>" for index in index_list])
    tgt_list = _merge_consecutive_elements(['<mask>' if not index in mask_list else src_str_list[index] for index in index_list])
    src_txt = _replace_mask(" ".join(src_list))
    tgt_txt = _replace_mask(" ".join(tgt_list))
    return src_txt,tgt_txt

if __name__ == "__main__":
    print(make_mask_textpair("A B C."))