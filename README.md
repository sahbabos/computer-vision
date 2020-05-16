# Face morphing computer-vision

This is simple classical application of affain matrix. For this project I simply use affain matrix to morph one face to another over 45 frames.


![](gif.gif)


## The procedures:
- Get some points from each image
- Use Delaunay fucntion to create the triangles trangulation 


### Get some points:
The basic Idea is to pick some points from one image and pick the correspondence points from the other.
Now you can use some library to do that for you I just wanted to doit with hand and curse myself afterwards.

```python
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


# print(p)

def trangulation(points_data_1, points_data_2):
	image_trangle_1 = Delaunay(np.array(points_data_1))
	image_trangle_2 = Delaunay(np.array(points_data_2))

	return image_trangle_1, image_trangle_2
```
![](3.png)
![](4.png)
