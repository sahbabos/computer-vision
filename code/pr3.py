import matplotlib.pyplot as plt
import matplotlib.image as image
from scipy.spatial import Delaunay
from skimage.draw import polygon
from statistics import mean
import imageio
import skimage.io as io
from scipy import misc
import numpy as np
import os
points_image_1 = []
points_image_2 = []
def getpoints(image,filename, points_array):
    plt.imshow(image)
    
    i = 1
    while True:
        x , y = plt.ginput(1, timeout = -1, mouse_add = 1)[0]
        points_array.append([x, y])
        plt.scatter(x, y)
        plt.draw()
        plt.text(x, y,i)
        i+=1
        if (i == 49):
            plt.savefig(filename + '_points.png')
            break
    plt.close()


def trangulation(points_data_1, points_data_2):
	image_trangle_1 = Delaunay(np.array(points_data_1))
	image_trangle_2 = Delaunay(np.array(points_data_2))

	return image_trangle_1, image_trangle_2


#show and save the triangles on the faces
def image_trangle_show(img, image_points, Delaunay_obj,filename):
	#this code is from https://doc;;s.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.spatial.Delaunay.html
	
	plt.triplot(image_points[:,0], image_points[:,1], Delaunay_obj.simplices.copy())
	plt.plot(image_points[:,0], image_points[:,1], 'd')
	plt.imshow(img)
	plt.show()
	plt.savefig(filename + '_tirangles.png')
	plt.close()

#this was tricky because i thought tri2 is between image 1 and image 2 but now i implement it to the midway image 
def computeAffine(tri1_pts, tri2_pts):
	# transformation matrix
	matrix_t = []

	m = np.zeros( (6, 6) )
	m[0][2], m[1][5], m[2][2], m[3][5], m[4][2], m[5][5] = 1, 1, 1, 1, 1, 1


	
	#element by element we find the right transformation rows and cols
	for tri_1, tri_2 in zip(tri1_pts, tri2_pts):
		m[0][0], m[0][1], m[1][3], m[1][4] = tri_1[0][0], tri_1[0][1],	tri_1[0][0], tri_1[0][1]
		m[2][0], m[2][1], m[3][3], m[3][4] = tri_1[1][0], tri_1[1][1],	tri_1[1][0], tri_1[1][1]
		m[4][0], m[4][1], m[5][3], m[5][4] = tri_1[2][0], tri_1[2][1],	tri_1[2][0], tri_1[2][1]
		vector = np.array([tri_2[0][0], tri_2[0][1], tri_2[1][0], tri_2[1][1], tri_2[2][0], tri_2[2][1]])

		matrix_t.append(np.vstack((np.reshape(np.linalg.lstsq(m, vector.T)[0],(2,3)), [0, 0, 1])))
	
	return matrix_t


def findmidface(image, points_1,points_2, weight):
	trasnition = np.zeros(image.shape)
	points_1 = np.array(points_1)
	points_2 = np.array(points_2)
	mid_shape = np.around(((1 - weight) * points_1) + (weight * points_2))
	mid_tri = Delaunay(mid_shape)
	
	del_obj_1, del_obj_2 = trangulation(points_1, points_2)

	#original image points for image 1
	oip_1 = points_1[mid_tri.simplices].copy()

	#destination image points for image 1
	dip_1 = mid_shape[mid_tri.simplices].copy()

	
	# #original image points for image 2
	# oip_2 = points_2[del_obj_2.simplices].copy()

	# #destination image points for image 2
	# dip_2 = mid_shape[mid_tri.simplices].copy()

	#finding the affaine matrix for each image to get the avrage points	
	m_1 = computeAffine(oip_1, dip_1)

	#m_1 = np.asarray(m_1)
	#this part if the fucntion warp the image to the new destination
	for orig, dest, solution in zip(oip_1, dip_1, m_1):
		mask = np.zeros(image.shape)
		dest = dest.T
		row, col = polygon(np.clip(dest[1].astype(int),0,image.shape[0]), np.clip(dest[0].astype(int),0,image.shape[1]))
		mask[row.astype(int),col.astype(int)] = 1
		y, x = np.where(mask[:,:,0])
		ones = np.ones(y.shape)
		tran_points = np.vstack([x, y, ones]).astype(int)
		rgb = np.dot(np.linalg.inv(solution),tran_points).astype(int)
		rgb[1] = np.clip(rgb[1],0,image.shape[0])
		rgb[0] = np.clip(rgb[0],0,image.shape[1])
		trasnition[tran_points[1], tran_points[0],:] = image[rgb[1], rgb[0],:]
	return trasnition
# def method_2(points_1, points_2, trai):
# 	trasnition = np.zeros(image.shape)
# 	points_1 = np.array(points_1)
# 	points_2 = np.array(points_2)

# 	transf = computeAffine(targe, source)
# 	for i in range(len(triangles)):



#         targe = points_1[trai.simplices[i]]
#         source = points_2[trai.simplices[i]]

        
#         poly = polygon(targe[:, 0], targe[:, 1])
#         polyRes = polygon(targetTriangle[:, 0], targetTriangle[:, 1])
#         Aline = np.vstack((polyRes[0], polyRes[1]))

#         Xline = np.dot(poly, np.vstack((Aline, np.ones(Aline.shape[1]))))[:2]



def morph(im1, im2, im1_pts, im2_pts, tri, warp_frac, dissolve_frac):
    trasnition_1 = np.zeros(im1.shape)
    trasnition_2 = np.zeros(im2.shape)
	
    mid_shape = np.around(((1 - warp_frac) * im1_pts) + (warp_frac * im2_pts))
    oip_1 = im1_pts[tri.simplices].copy()

	#destination image points for image 1
    dip_1 = mid_shape[tri.simplices].copy()
  

    m_1 = computeAffine(oip_1, dip_1)

	#m_1 = np.asarray(m_1)
	#this part if the fucntion warp the image to the new destination
  

    for orig, dest, solution in zip(oip_1, dip_1, m_1):
    	mask = np.zeros(im1.shape)
    	dest = dest.T
    	row, col = polygon(np.clip(dest[1].astype(int),0,im1.shape[0]), np.clip(dest[0].astype(int),0,im1.shape[1]))
    	mask[row.astype(int),col.astype(int)] = 1
    	y, x = np.where(mask[:,:,0])
    	ones = np.ones(y.shape)
    	tran_points = np.vstack([x, y, ones]).astype(int)
    	rgb = np.dot(np.linalg.inv(solution),tran_points).astype(int)
    	rgb[1] = np.clip(rgb[1],0,im1.shape[0])
    	rgb[0] = np.clip(rgb[0],0,im1.shape[1])
    	trasnition_1[tran_points[1], tran_points[0],:] = im1[rgb[1], rgb[0],:]
    oip_2 = im2_pts[tri.simplices].copy()
    dip_2 = mid_shape[tri.simplices].copy()
    m_2 = computeAffine(oip_2, dip_2)
    for orig, dest, solution in zip(oip_2, dip_2,m_2):
    	mask = np.zeros(im2.shape)
    	dest = dest.T
    	row, col = polygon(np.clip(dest[1].astype(int),0,im2.shape[0]), np.clip(dest[0].astype(int),0,im2.shape[1]))
    	mask[row.astype(int),col.astype(int)] = 1
    	y, x = np.where(mask[:,:,0])
    	ones = np.ones(y.shape)
    	tran_points = np.vstack([x, y, ones]).astype(int)
    	rgb = np.dot(np.linalg.inv(solution),tran_points).astype(int)
    	rgb[1] = np.clip(rgb[1],0,im2.shape[0])
    	rgb[0] = np.clip(rgb[0],0,im2.shape[1])
    	trasnition_2[tran_points[1], tran_points[0],:] = im2[rgb[1], rgb[0],:]

    return ((1 - dissolve_frac) * trasnition_1) + (dissolve_frac * trasnition_2)
## this code was used from Sean Farhat and its not my code
##start
def parse_asf(filename, width, height):
   
    with open(filename, "r") as f:

        data = f.read()
        lines = data.split("\n")
        num_points = int(lines[9])
        points = [line.split("\t") for line in lines[16:16+num_points]]
        unformatted = [pt[2:4] for pt in points]
        decimal_xy = [[float(point[0]), float(point[1])] for point in unformatted]

        xy = [[x * width, y * height] for x, y in decimal_xy]
        xy.append([0, 0])
        xy.append([width, 0])
        xy.append([0, height])
        xy.append([width, height])

        return xy


##end
def cal_average(num):
    sum_num = np.array(num[0])
    for t in range(1,len(num)):
        sum_num = sum_num + np.array(t)           

    avg = np.array(sum_num) / len(num)
    return avg

def mean_face_population():
	index=['01','02','03','04','05','06','07','09','10','11','13','16','17','18','19','20','21','23','24','25','26','27','28','29','31','32','33','34','36','37','38','39','40']
	data_folder = "imm_face_db/"
	w = 640
	h = 480
	filename = '-1m.jpg'
	pointname = '-1m.asf'
	img = []
	p = []
	mid_face = np.zeros((h,w,3))
	list_of_avg_face = []
	for i in index:
		img.append(io.imread(data_folder + str(i)+filename))
		p.append(parse_asf(data_folder + str(i) + pointname,w,h))
	avragepoint = cal_average(p)
	mid_tri = Delaunay(avragepoint)
	first_img =  img [0]
	first_p = p[0]
	j = 0
	for i in range(1,len(img)):
		
		points_1 = np.asarray(p[i])
		new_image_1 = findmidface(img[i], points_1, avragepoint, .5) /2
		if i < 2 :
			new_image_1 = findmidface(img[0], p[0], avragepoint, .5) /2
			new_image_2 = findmidface(img[1], p[1], avragepoint, .5) /2
			mid_face= (img[0]+img[1] ) /2
		else:
			new_image_1 = findmidface(img[i], p[i], avragepoint, .5) /2
			mid_face= (img[0]+mid_face ) /2


	return mid_face

def mean_with_myface(myface, points, mean_face):
	index=['01','02','03','04','05','06','07','09','10','11','13','16','17','18','19','20','21','23','24','25','26','27','28','29','31','32','33','34','36','37','38','39','40']
	data_folder = "imm_face_db/"
	w = 640
	h = 480
	filename = '-1m.jpg'
	pointname = '-1m.asf'
	img = []
	p = []
	mid_face = np.zeros((h,w,3))
	list_of_avg_face = []
	for i in index:
		img.append(io.imread(data_folder + str(i)+filename))
		p.append(parse_asf(data_folder + str(i) + pointname,w,h))
	avragepoint = cal_average(p)
	mid_shape = np.around(((1 - .5) * points) + (.5 * avragepoint))
	new_image_1 = findmidface(myface, points, mid_shape, .5) /2
	new_image_2 = findmidface(mean_face, avragepoint, mid_shape, .5) /2

	return findmidface(myface, mean_face, mid_shape, .5) /2

def main ():
	#uncomment for each part of the project
	# #load the Images
	im_1 = io.imread("sahba1.jpg",plugin='pil')
	im_2 = io.imread("sepehr2.jpg",plugin='pil')

	
	# #get the points for each image I collected 40 points from the figure and 4 from the corners
	getpoints(im_1, "sahba1", points_image_1)
	# getpoints(im_2, "sepehr2", points_image_2)
	# # using the delaunay to find the right trangulations for each image and return the trangles  
	
	# del_obj_1, del_obj_2 = trangulation(points_image_1,points_image_2)

	# #now lets plot them and save the plot
	# image_trangle_show(im_1, np.array(points_image_1), del_obj_1, "sahba1.jpg")
	# image_trangle_show(im_2, np.array(points_image_2), del_obj_2, "sepehr2.jpg")
	points_1 = points_image_1.copy()
	# points_2 = points_image_2.copy()
	points_1 = np.array(points_1)
	# points_2 = np.array(points_2)
 
	# mid_shape = np.around(((1 - .5) * points_1) + (.5 * points_2))
	# mid_tri = Delaunay(mid_shape)
	# new_image_1 = findmidface(im_1, points_image_1, points_image_2, .5) /2
	# new_image_2 = findmidface(im_2, points_image_2, points_image_1, .5) /2
	#mid_face = np.add(new_image_1 , new_image_2[:new_image_1.shape[0],:new_image_1.shape[1],:])
	#io.imsave('test.png',mid_face)


	#uncomment to do the morth just change the name of the file you want to morph above
	##### morphing:
	# weights = np.linspace(0.0, 1.0, 45)
	# mid_tri = Delaunay(mid_shape)
	# gif = []
	# count = 1
	# for i in weights:
	# 	morph_image = morph(im_1, im_2, points_1, points_2, mid_tri, i, i)
	# 	io.imsave('image_'+ str(count) +  '.png',morph_image)
	# 	gif.append(morph_image)
	# 	count+=1 
	# gif.save('morph.gif',format='GIF', append_images=gif[1:], save_all =True, duration=300, loop = 0)
	
	#find the avrage face of dena population
	mean = mean_face_population()
	io.imsave('mean_face.png',mean)

	myface = mean_with_myface(im_1, points_1, mean)
	io.imsave('my_face.png',myface)
if __name__ == "__main__":
	main()
