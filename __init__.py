import matplotlib.pyplot as plt
from fcm import FCM
import cv2
import numpy as np

if __name__ == "__main__":
    img = cv2.imread("colored_image.png",1)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    z=img.reshape(-1,3)
    z=np.float32(z)
    #a= z.reshape(229,459,3)
    x=input("enter number of clusters: ")
    fcm=FCM(z,x,100) #create the fcm model
    fcm.initializeCenters() #initialize the centers of clusters
    current=0
    while((current<fcm.max_iter) and(not fcm.membershipConvergence())): #while the membership values not converging
        print(current)
        fcm.updateMembershipDegrees() 
        fcm.updateCenters()      
        current+=1
    output1 = fcm.segmentImage()
    output1 = np.array(output1)
    output1 = output1.reshape(229,459,3)
    output1 = np.uint8(output1)
    output = [img, output1]
    titles = ["original", str(x)+" clusters segmentation"]
    for i in range(2):
        plt.subplot(2,2,i+1)
        plt.imshow(output[i])
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])
    plt.show()