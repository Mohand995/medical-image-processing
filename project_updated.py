import tkinter as tk 
from tkinter import filedialog as fd
from tkinter import LabelFrame
from PIL  import Image 
from PIL import ImageTk 
import cv2 
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
import skimage 

def select_image():
   global panelA
   global path
   path=fd.askopenfilename();
   if len(path) > 0: ##to make sure that a file is selected and we didn't choose cancel
       img = cv2.imread(path,0) ##read image
       img=cv2.resize(img,(300,300))
       ##to display image on tkinter first we change format
       img = Image.fromarray(img) ##convert to PIL format
       img = ImageTk.PhotoImage(img) ##convert to TK format
       
       if panelA is None : ##if panel is none show original image
         panelA = tk.Label(image=img)
         panelA.image = img ##without it python will load a garbage and delete the img
         panelA.pack(side="right", padx=10, pady=10)
       else:
         panelA.configure(image=img) ##if image is loaded upload it's configuration
         panelA.image = img
         return path
 
def  apply_gaussian_noise(p):
        img = cv2.imread(p); 
        global noisy
        noisy=skimage.util.random_noise(img,mode="gaussian",var=0.01) ;
        cv2.imshow("gaussian noise",noisy); 
        cv2.waitKey(0); 
        cv2.destroyAllWindows(); 
        
 
def  apply_salt_pep_noise(p):
        img = cv2.imread(p); 
        global noisy
        noisy=skimage.util.random_noise(img,mode='s&p',amount=0.3) ;
        cv2.imshow("salt and pep",noisy); 
        cv2.waitKey(0); 
        cv2.destroyAllWindows();
        
def  contrast_Enhance(p):
        img = cv2.imread(p); 
        #add two images with fusion weghiting as the first is our img and 
        #second is black img with same dimension and width
        #dst = src1 * alpha + src2 * beta + gamma
        #alpha less than 1 low contrast more than 1 high cont
        #src2 for brightness
        result = cv2.addWeighted(img,2,np.zeros(img.shape,img.dtype),0,-1)
        cv2.imshow("contrast",result); 
        cv2.waitKey(0); 
        cv2.destroyAllWindows(); 
   
def  zoom(p):
        img = cv2.imread(p); 
        ##inter_linear used for zooming
        result = cv2.resize(img,None,fx=0.6*15,fy=0.6*15,interpolation=cv2.INTER_LINEAR)
        cv2.imshow("zoom",result); 
        cv2.waitKey(0); 
        cv2.destroyAllWindows(); 
       
def  remove_noise(p):
        global panelB
        f= ndimage.median_filter(noisy, 5);
        cv2.imshow("median",f); 
        image_gaussian = ndimage.gaussian_filter(noisy, 5)
        cv2.imshow("gaussian",image_gaussian);
        meanBlurKernel = np.ones((3, 3), np.float32)/9
        meanBlur = cv2.filter2D(src=noisy, kernel=meanBlurKernel, ddepth=-1)
        cv2.imshow("mean",meanBlur);
        sharpenKernel = np.array(([[0, -1, 0], [-1, 9, -1], [0, -1, 0]]), np.float32)/9
        sharpen = cv2.filter2D(src=noisy, kernel=sharpenKernel, ddepth=-1)
        cv2.imshow("sharpening",sharpen);
        cv2.waitKey(0); 
        cv2.destroyAllWindows();
        
def canny(p):
    global panelB
    img = cv2.imread(p);
    img=cv2.resize(img,(300,300))
    ##to apply canny we first make gaussian filter wit 5x5 for ex then we
    ## pass at all pixels to make gradient magnitude and direction 
    ## then remove pixels that not considerd edges and search for local max
    ## then we define minval and maxval if point exceed them they are edge points
    ## and points between two threshold are classified according to connectivity
    c= cv2.Canny(img,150,255)
    c= Image.fromarray(c)
    c= ImageTk.PhotoImage(c)
    if panelB is None :
     panelB = tk.Label(image=c)
     panelB.image = c
     panelB.pack(side="left", padx=10, pady=10)
    else:
          panelB.configure(image=c)
          panelB.image = c
          
def sobel(p):
    global panelB
    img = cv2.imread(p,0); 
    img=cv2.resize(img,(300,300))
    #The Sobel Operator is a discrete differentiation operator. It computes an approximation of the gradient of an image intensity function.
    #The Sobel Operator combines Gaussian smoothing and differentiation.
    c = cv2.Sobel(img ,cv2.CV_8U,0,1,ksize=3)  ##unsigned 8 bit per pixel for display
    c= Image.fromarray(c)
    c= ImageTk.PhotoImage(c)
    if panelB is None :
     panelB = tk.Label(image=c)
     panelB.image = c
     panelB.pack(side="left", padx=10, pady=10)
    else:
          panelB.configure(image=c)
          panelB.image = c
    
def rotat(p):
    img = cv2.imread(p);
    img_rotate_90_clockwise = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    cv2.imshow("rotated",img_rotate_90_clockwise)
    cv2.waitKey(0); 
    cv2.destroyAllWindows(); 
    


# mouse callback function
def HIST(p):
    img = cv2.imread(p,0) 
    plt.hist(img.ravel(),256,[0,256])
    plt.title('Histogram for gray scale picture')
    plt.show()
# apply histogram equalization to image
    equ = cv2.equalizeHist(img)   
# stacking images side-by-side
    cv2.imshow("aa",equ)    
    cv2.waitKey(0); 
    cv2.destroyAllWindows();
    return equ
    
def detect(p):
    img = cv2.imread(p,0) 
    img=cv2.resize(img,(300,300))
    global panelB
    img= ndimage.median_filter(img, 5);
    ##pixels over 120 is 1 and below 120 is 0
    ret,t = cv2.threshold(img,120,255,cv2.THRESH_BINARY)

    t= Image.fromarray(t)
    t= ImageTk.PhotoImage(t)
    if panelB is None :
          panelB = tk.Label(image=t)
          panelB.image = t
          panelB.pack(side="left", padx=10, pady=10)
    else:
           panelB.configure(image=t)
           panelB.image = t
         
def different_thresholding(p):
    
    img = cv2.imread(p,0) 
    ret,thresh1_img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    ret,thresh2_img = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
    ret,thresh3_img = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
    ret,thresh4_img = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
    ret,thresh5_img = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)

    titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
    images = [img, thresh1_img, thresh2_img, thresh3_img, thresh4_img, thresh5_img]

    for i in range(6):
        plt.subplot(2,3,i+1),plt.imshow(images[i],cmap = 'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
        plt.show()
          
root = tk.Tk()
panelA = None
panelB = None
panelC=None
# create a button, then when pressed, will trigger a file chooser
# dialog and allow the user to select an input image; then add the
# button the GUI
k=LabelFrame(root,text="show image")
k.pack(fill="both",padx=40,pady=1)
a=LabelFrame(root,text="Edge detect")
a.pack(fill="both",padx=40,pady=1)
b=LabelFrame(root,text="Apply noise")
b.pack(fill="both",padx=40,pady=1)
r=LabelFrame(root,text="filter noise")
r.pack(fill="both",padx=40,pady=1)
w=LabelFrame(root,text="Histogram")
w.pack(fill="both",padx=40,pady=1)
v=LabelFrame(root,text="rotate")
v.pack(fill="both",padx=40,pady=1)
l=LabelFrame(root,text="zoom")
l.pack(fill="both",padx=40,pady=1)
y=LabelFrame(root,text="contrast enhance")
y.pack(fill="both",padx=40,pady=1)

btn = tk.Button(k, text="Browse image", command=select_image)
btn.pack(side="bottom")
btn2 = tk.Button(b, text="apply gaussian noise", command=lambda:apply_gaussian_noise(path))
btn2.pack(side="bottom")
btn3 = tk.Button(r, text="filter noise", command=lambda:remove_noise(path))
btn3.pack(side="bottom")
btn5 = tk.Button(a, text="Canny", command=lambda:canny(path))
btn5.pack(side="bottom")
btn6 = tk.Button(v, text="rotate", command=lambda:rotat(path))
btn6.pack(side="bottom")
btn7 = tk.Button(w, text="Hist equalization", command=lambda:HIST(path))
btn7.pack(side="bottom")
btn8 = tk.Button(a, text="Detect", command=lambda:detect(path))
btn8.pack(side="bottom")
btn9 = tk.Button(a, text="sobel", command=lambda:sobel(path))
btn9.pack(side="bottom")
btn10 = tk.Button(y, text="contrast_enhance", command=lambda:contrast_Enhance(path))
btn10.pack(side="bottom")
btn11 = tk.Button(l, text="zoom", command=lambda:zoom(path))
btn11.pack(side="bottom")
btn12 = tk.Button(b, text="apply s&p noise", command=lambda:apply_salt_pep_noise(path))
btn12.pack(side="bottom")
btn13 = tk.Button(a, text="different thresholding", command=lambda:different_thresholding(path))
btn13.pack(side="bottom")
# kick off the GUI
root.mainloop()     
			
	    
		





