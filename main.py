import cv2
import numpy
from matplotlib import pyplot as plt
from matplotlib import animation as anim

def grabFrame(vc):
    ret, frame = vc.read()
    return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ret else None)

def processFrame(frame):
    # Convert to grayscale
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Create a blurred images to smooth it out
    blur = cv2.medianBlur(img, 5)

    # Create a mask that includes only bright white elements
    ret, mask = cv2.threshold(blur, 253, 255, cv2.THRESH_BINARY)

    # Look for outlines in the mask image
    edges = cv2.Canny(mask, 100, 200)

    circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, 1, 20, param1=30, param2=20, minRadius=3, maxRadius=60)

    if circles is not None:
        circles = numpy.uint16(numpy.around(circles))

        for i in circles[0, :]:
            cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)
    
    return [frame, blur, edges, mask]

def update(j, vc, aimgs): 
    ret, frame = grabFrame(vc)   
    if (not ret):
        return

    images = processFrame(frame)
    for i in range(len(images)):
        aimgs[i].set_data(images[i])

def circles():
    print("Reading video file")
    vc = cv2.VideoCapture("images/wand_cross1.mov")
    if (not vc.isOpened()):
        print("Could not read video file")
        exit(1)

    figure, subplots = plt.subplots(2, 2)
    titles = ['frame', 'blur', 'edges', 'mask']

    ret, frame = grabFrame(vc)
    if (not ret):
        print("Could not grab first frame. Exiting...")
        exit(1)

    images = processFrame(frame)
    aimgs = []
    for i in range(len(images)):
        plt.subplot(2, 2, i+1)
        aimg = plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])
        aimgs.append(aimg)

    f = anim.FuncAnimation(figure, update, interval=20, fargs=(vc, aimgs))

    plt.show()

def image():
    print("Reading image file")
    img = cv2.imread("images/wand2.jpg", cv2.IMREAD_GRAYSCALE)
    if img.data == None:
        print("Could not open image file. Exiting...")
        exit(1)

    print("Showing image in window")
    cv2.imshow('wand1', img)

    print("Waiting for keyboard input...")
    cv2.waitKey(0)

    print("Destroying all windows")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Running...")
    circles()
    print("Exiting...")