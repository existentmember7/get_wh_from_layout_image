from PIL import Image, ImageDraw
import numpy as np
import math
import cv2
from numpy.core.fromnumeric import shape
from tqdm import tqdm, trange
import argparse

def find_column(img):
    index = 0
    count_list = np.zeros(len(img))
    cor_list = np.zeros((len(img),2),dtype=int)
    end_cor_list = np.zeros((len(img),2),dtype=int)
    print("Find Column...")
    for i in tqdm(range(len(img))):
        start = False
        start_y = 0
        count = 0
        for j in range(len(img[i])):
            if img[i][j] == 225 and start == False:
                start = True
                start_y = j
                count += 1
            elif img[i][j] == 225 and start:
                count += 1
            elif img[i][j] != 225 and start == False:
                pass
            elif img[i][j] != 225 and start:
                if count > count_list[index]:
                    count_list[index] = count
                    cor_list[index][0] = i
                    cor_list[index][1] = start_y
                    end_cor_list[index][0] = i
                    end_cor_list[index][1] = j
                count = 0
                start = False
        index += 1

    # temp_count_list = np.sort(count_list, axis=0)

    return count_list, cor_list, end_cor_list

def find_brench(img, count_list, cor_list, direction):

    angles =  math.pi*np.arange(1.0,46.0,1.0)/180
    potential_list = np.zeros(len(img))
    potential_cor_list = np.zeros((len(img),2))
    print("Find brench", direction, "...")
    for i in tqdm(range(0, len(count_list))):
        index = 0
        start_index = i
        start_point = cor_list[start_index]
        count_list_1 = np.zeros(len(angles))
        cor_list_1 = np.zeros((len(angles), 2))
        for angle in angles:
            x = direction[0] * 1
            y = direction[1] * math.tan(angle)
            
            start = False
            
            p_x = start_point[0]
            p_y = start_point[1] - direction[1] * 10
            temp_p_x = p_x
            temp_p_y = p_y
            count = 0
            while True:
                
                temp_p_x += x
                temp_p_y += y

                p_x = int(temp_p_x)
                p_y = int(temp_p_y)
                if p_x < 0 or p_y < 0 or p_x >= img.shape[0] or p_y >= img.shape[1]:
                    break
                if img[p_x][p_y] == 225 and start == False:
                    start = True
                    count += (x**2 + y**2)**0.5
                elif img[p_x][p_y] == 225 and start:
                    count += (x**2 + y**2)**0.5
                elif img[p_x][p_y] != 225 and start == False:
                    pass
                elif img[p_x][p_y] != 225 and start:
                    if count_list_1[index] < count:
                        count_list_1[index] = count + 2 * count_list[start_index]
                        cor_list_1[index][0] = p_x
                        cor_list_1[index][1] = p_y
                    count = 0
                    start = False
                 
            index += 1

            temp_count_list_1 = np.sort(count_list_1, axis=0)
            potential_list[start_index] = count_list_1[np.where(count_list_1 == temp_count_list_1[-1])[0][0]]
            potential_cor_list[start_index] = cor_list_1[np.where(count_list_1 == temp_count_list_1[-1])[0][0]]

    temp_potential_list = np.sort(potential_list)
    # show_result(np.where(potential_list == temp_potential_list[-1]), cor_list, potential_cor_list, potential_list)
    # draw_line(img, np.where(potential_list == temp_potential_list[-1]), cor_list, potential_cor_list, direction)
    return np.where(potential_list == temp_potential_list[-1]), potential_cor_list, direction

def draw_line(img, index, cor_list, potential_cor_list, direction):
    shape = [(cor_list[index][0][0], cor_list[index][0][1] - direction[1] * 10), (potential_cor_list[index][0][0], potential_cor_list[index][0][1])]
    img1 = ImageDraw.Draw(img)
    img1.line(shape, fill = "green", width = 3)
    print("brench: ", shape)
    # img.show()
    return img, shape

def draw_column(img, cor_list, end_cor_list, indexes, directions):
    start = [0,0]
    end = [0,0]
    for i in range(len(indexes)):
        if directions[i][1] == -1:
            start[0] += cor_list[indexes[i]][0][0]
            start[1] += cor_list[indexes[i]][0][1]
        elif directions[i][1] == 1:
            end[0] += end_cor_list[indexes[i]][0][0]
            end[1] += end_cor_list[indexes[i]][0][1]
    start[0] = start[0]/2
    start[1] = start[1]/2
    end[0] = end[0]/2
    end[1] = end[1]/2
    shape = [(start[0], start[1] + 10), (end[0], end[1] - 10)]
    print("column: ", shape)
    img1 = ImageDraw.Draw(img)
    img1.line(shape, fill = "green", width = 3)
    # img.show()
    return img, shape

def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	# return the warped image
	return warped

def show_result(index, cor_list, potential_cor_list, potential_list):
    print(cor_list[index])
    print(potential_cor_list[index])
    print(potential_list[index])

def calculate_length(y, pix_y, pix_x):
    x = y * pix_x/pix_y
    return x


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "path to the image file")
args = vars(ap.parse_args())

def main():
    img = Image.open(args["image"])
    width, height = img.size
    img = img.crop((int(width*0.1), 0, int(width*0.9), height))
    img = np.array(img)
    img = img.swapaxes(0,1)
    print(img.shape)

    indexes = []
    potantial_cor_lists = []
    directions = []

    count_list, cor_list, end_cor_list = find_column(img)

    for i in [1,-1]:
        for j in [1,-1]:
            if j == -1:
                index, potantial_cor_list, direction = find_brench(img, count_list, cor_list, (i, j))
                indexes.append(index)
                potantial_cor_lists.append(potantial_cor_list)
                directions.append(direction)
            elif j == 1:
                index, potantial_cor_list, direction = find_brench(img, count_list, end_cor_list, (i, j))
                indexes.append(index)
                potantial_cor_lists.append(potantial_cor_list)
                directions.append(direction)
    
    l_pts = []
    r_pts = []
    img = Image.fromarray(img.swapaxes(0,1))
    for line in range(len(indexes)):
        if directions[line][1] == -1:
            img, pt = draw_line(img, indexes[line], cor_list, potantial_cor_lists[line], directions[line])
            if directions[line][0] == -1:
                l_pts.append(pt[1])
            elif directions[line][0] == 1:
                r_pts.append(pt[1])
        elif directions[line][1] == 1:
            img, pt = draw_line(img, indexes[line], end_cor_list, potantial_cor_lists[line], directions[line])
            if directions[line][0] == -1:
                l_pts.append(pt[1])
            elif directions[line][0] == 1:
                r_pts.append(pt[1])

    img, pt = draw_column(img, cor_list, end_cor_list, indexes, directions)
    l_pts.append(pt[0])
    l_pts.append(pt[1])
    r_pts.append(pt[0])
    r_pts.append(pt[1])
    img.save("frame.png")

    # load the image and grab the source coordinates (i.e. the list of
    # of (x, y) points)
    # NOTE: using the 'eval' function is bad form, but for this example
    # let's just roll with it -- in future posts I'll show you how to
    # automatically determine the coordinates without pre-supplying them
    image = np.array(img)
    l_pts = np.array(l_pts, dtype = "float32")
    r_pts = np.array(r_pts, dtype = "float32")
    # apply the four point tranform to obtain a "birds eye view" of
    # the image
    l_wraped = four_point_transform(image, l_pts)
    r_wraped = four_point_transform(image, r_pts)
    print(l_wraped.shape)
    print(r_wraped.shape)
    cv2.imwrite("l_transform.png", l_wraped)
    cv2.imwrite("r_transform.png", r_wraped)
    lx = calculate_length(3, l_wraped.shape[0], l_wraped.shape[1])
    rx = calculate_length(3, r_wraped.shape[0], r_wraped.shape[1])
    print("left_x: ", lx)
    print("right_x: ", rx)
    



if __name__ == "__main__":
    main()
