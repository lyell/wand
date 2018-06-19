import cv2
import numpy
from matplotlib import pyplot as plt
from matplotlib import animation as anim

p = []

def grabFrame(vc):
    ret, frame = vc.read()
    return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ret else None)

def processFrame(frame):
    # Convert to grayscale
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Create a blurred images to smooth it out
    blur = cv2.medianBlur(img, 7)

    # Create a mask that includes only bright white elements
    ret, mask = cv2.threshold(blur, 254, 255, cv2.THRESH_BINARY)

    # Look for edges in the mask image
    imgc, contours, hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Draw the contours on the image
    cv2.drawContours(frame, contours, -1, (255, 0, 0), 2)

    # Find the centroid of each contour
    for i in range(len(contours)):
        c = contours[i];
        m = cv2.moments(c)

        if m['m00'] > 0:
            cx = int(m['m10']/m['m00'])
            cy = int(m['m01']/m['m00'])

            # Draw the path the centroid has taken
            # TODO: store the trail for each centroid in a separate array
            # p.append([cx, cy])
            # np = numpy.array(p)
            #if len(p) >= 3:
            #    cv2.polylines(frame, numpy.int32([np]), False, (0, 0, 255), 2)

            # Draw the centroid on the image
            cv2.circle(frame, (cx, cy), 2, (0, 255, 0), 2)

    return [frame, blur, imgc, mask]

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
    titles = ['frame', 'blur', 'contours', 'mask']

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