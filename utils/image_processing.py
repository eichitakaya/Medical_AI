import numpy as np

def divide(img_arr, size) -> np.array:
    # 縦横それぞれ割り切れるか確認し，余る分はpaddingする
    mod_row = img_arr.shape[0] % size
    mod_column = img_arr.shape[1] % size
    pad_row = size - mod_row
    pad_column = size - mod_column
    arr = np.pad(img_arr, [(0, pad_row), (0, pad_column)])
    
    # 行方向→列方向の順に分割
    rows = np.split(arr, int(arr.shape[0]/size))
    parts = []
    for row in rows:
        parts.append(np.split(row, int(arr.shape[1]/size), axis=1))
    
    parts_n = int((arr.shape[1]/size) * (arr.shape[0]/size))

    # partsの1行目をnp.arrayに変換し，2行目以降は追加していく
    return_parts = np.stack(parts[0])
    for i in range(len(parts)-1):
        return_parts = np.vstack([return_parts, parts[i+1]])
    
    row_n = len(parts)
    column_n = int(parts_n / row_n)
    return return_parts, (row_n, column_n)

def synthesize(parts, row_column):
    row_list = []
    row_n, column_n = row_column

    for i in range(row_n):
        column_list = []
        for j in range(column_n):
            column_list.append(parts[i * column_n + j])
            #print(i*row_n + j)
        row_list.append(np.concatenate(column_list, axis=1))
    synthesized = np.concatenate(row_list)
    return synthesized

if __name__ == '__main__':
    from PIL import Image
    img_arr = np.array(Image.open("rectangle.png").convert("L"))
    print(img_arr.shape)
    parts, row_column = divide(img_arr, 100)
    print(parts.shape)
    img_parts = Image.fromarray(parts[9])
    img_parts.save("parts.png")
    synthesized =  synthesize(parts, row_column)
    img_synthesized = Image.fromarray(synthesized)
    img_synthesized.save("synthesized.png")
    print(synthesized.shape)