def contours2list(contours):
    point_list = []
    for i in range(contours[0].shape[0]):
        point_list.append(tuple(contours[0][i][0]))
    return point_list

def window(img):
    ww = img.max() - img.min()
    wc = int(ww / 2)
    
    window_max = wc + ww / 2
    window_min = wc - ww / 2
    
    img = 255 * (img - window_min) / (window_max - window_min)
    img[img > 255] = 255
    img[img < 0] = 0
    return img

dcm = pydicom.read_file("0.dcm")
img = Image.fromarray(np.uint8(window(dcm.pixel_array)))
img.save("raw.png")

def superimpose(img_path, label_path, save_path):
    img = Image.open(img_path)
    img = img.convert("RGBA")
    mask = np.array(Image.open(label_path))
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    poly = Image.new('RGBA', img.size)
    pdraw = ImageDraw.Draw(poly)
    point_list = contours2list(contours)
    pdraw.polygon(point_list, fill=(255,0,255,128), outline=(0,0,0,255))
    img.paste(poly,mask=poly)
    img.save(save_path)
