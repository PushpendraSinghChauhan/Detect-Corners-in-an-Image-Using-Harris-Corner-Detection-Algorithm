import sys
import cv2
import numpy as np

if __name__ == '__main__':

    if(len(sys.argv) == 3):
        input_file = sys.argv[1]
        input_file1 = sys.argv[2]
        filename = "../data/" + input_file
        filename1 = "../data/" + input_file1
        image = cv2.imread(filename)
        image1 = cv2.imread(filename1)
        if (image is None or image1 is None):
            print("File with name - \'"+input_file+"\' or \'"+input_file1+"\' does not exist in data folder .")
            inp_file = input("PRESS - 'C' to capture image using Camera  OR  Press - 'D' to continue with default image ")
            if(inp_file == 'D'):
                filename = "../data/square2.jpg"
                image = cv2.imread(filename)
                filename1 = "../data/square2.jpg"
                image1 = cv2.imread(filename1)                
        
            elif(inp_file == 'C'):
                camera = cv2.VideoCapture(0)
                input("Press Enter to capture !")
                return_value, image = camera.read()
                # Handle large images
                if(image.shape[0]>400 or image.shape[1]>400):
                    filename = "../data/resizeImage.jpg"
                    image = cv2.resize(image,(400,400))
                    cv2.imwrite(filename, image)                
                camera.release()
                
                camera1 = cv2.VideoCapture(0)
                input("Press Enter to capture !")
                return_value, image1 = camera1.read()
                # Handle large images
                if(image1.shape[0]>400 or image1.shape[1]>400):
                    filename1 = "../data/resizeImage1.jpg"
                    image1 = cv2.resize(image1,(400,400))
                    cv2.imwrite(filename1, image1)                
                camera1.release()                
            
            else:
                sys.exit("Please input correct filename from command line again !") 
        
        else:
            # Handle large images
            if(image.shape[0]>400 or image.shape[1]>400):
                filename = "../data/resizeImage.jpg"
                image = cv2.resize(image,(400,400))
                cv2.imwrite(filename, image)
                
            # Handle large images
            if(image1.shape[0]>400 or image1.shape[1]>400):
                filename1 = "../data/resizeImage1.jpg"
                image1 = cv2.resize(image1,(400,400))
                cv2.imwrite(filename1, image1)
        
    else:
        print("No Image file name entered .")
        inp_file = input("PRESS - 'C' to capture image using Camera  OR  Press - 'D' to continue with default image ")
        if(inp_file == 'D'):
            filename = "../data/square2.jpg"
            image = cv2.imread(filename)
            filename1 = "../data/square2.jpg"
            image1 = cv2.imread(filename1)        
        
        elif(inp_file == 'C'):
            camera = cv2.VideoCapture(0)
            input("Press Enter to capture !")
            return_value, image = camera.read()
            # Handle large images
            if(image.shape[0]>400 or image.shape[1]>400):
                filename = "../data/resizeImage.jpg"
                image = cv2.resize(image,(400,400))
                cv2.imwrite(filename, image)                
            camera.release()
                
            camera1 = cv2.VideoCapture(0)
            input("Press Enter to capture !")
            return_value, image1 = camera1.read()
            # Handle large images
            if(image1.shape[0]>400 or image1.shape[1]>400):
                filename1 = "../data/resizeImage1.jpg"
                image1 = cv2.resize(image1,(400,400))
                cv2.imwrite(filename1, image1)                
            camera1.release()            
            
        else:
            sys.exit("Please input correct filename from command line again !") 
    

    
    
    # Function to apply Harris Corner Detection algorithm
    def apply_harris(img, window_size, weight_trace, threshold):
        
        # Gradient of image
        grad_x = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
        grad_y = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)        
        
        start = window_size//2
        # Correlation matrix C = [[sum(x_i^2), sum(x_i * y_i)], [sum(x_i * y_i),  sum(y_i^2)]]
        grad_x_squared = grad_x**2
        grad_y_squared = grad_y**2
        grad_xy = grad_x*grad_y
        
        corner_list = []
        
        # Finding corners using window
        for i in range(start+4, img.shape[0]-start-4):
            for j in range(start+4, img.shape[1]-start-4):
                # Calculate sum of window elements
                window_x_squared = grad_x_squared[i-start:i+start+1, j-start:j+start+1]
                window_xy = grad_xy[i-start:i+start+1, j-start:j+start+1]
                window_y_squared = grad_y_squared[i-start:i+start+1, j-start:j+start+1]
                
                sum_x_squared = window_x_squared.sum()
                sum_y_squared = window_y_squared.sum()
                sum_xy = window_xy.sum()
                
                # Formula to find cornerness measure : Cornerness(C) = determinant(C) - k * trace(C)^2
                # Here, C = correlation matrix and k = weight of trace
                determinant_C = (sum_x_squared*sum_y_squared) - (sum_xy**2)
                trace_C = sum_x_squared+sum_y_squared
                cornerness_measure = determinant_C - weight_trace * (trace_C**2)
                
                # If cornerness_measure is above threshold, add to the corner_list with value of [loc_j, loc_i, cornerness_measure]
                if(cornerness_measure > threshold):
                    corner_list.append([j, i, cornerness_measure])
                    
        refined_corner_list = []            
        end = img.shape[0]-(img.shape[0]%window_size) 
        end1 = img.shape[1]-(img.shape[1]%window_size)
        
      
       
        # Locating corners using window
        for i in range(0, end, window_size):
            for j in range(0, end1, window_size):
                window_corner = []
                for corner in corner_list:
                    if((corner[0]>=j and corner[0]<(j+window_size)) and (corner[1]>=i and corner[1]<(i+window_size))):
                        window_corner.append(corner)
                        
                corner_sum = []
                for corn in window_corner:
                    # Calculate sum of window elements
                    window_x_squared = grad_x_squared[i:i+window_size, j:j+window_size]
                    window_xy = grad_xy[i:i+window_size, j:j+window_size]
                    window_y_squared = grad_y_squared[i:i+window_size, j:j+window_size]
                
                    sum_x_squared = window_x_squared.sum()
                    sum_y_squared = window_y_squared.sum()
                    sum_xy = window_xy.sum()
                
                    #corr_matrix = np.array([[sum_x_squared,sum_xy],[sum_xy,sum_y_squared]])
                    #inv_corr = np.linalg.inv(corr_matrix)
                    #temp_arr = np.array([0,0])
                    elem_sum = 0
                    for x in range(i,i+window_size):
                        for y in range(j,j+window_size):
                            # (X_i - P)
                            diff_x = y-corn[0]
                            diff_y = x-corn[1]
                            temp_1 = (grad_x_squared[x,y]*diff_x+grad_xy[x,y]*diff_y)
                            temp_2 = (grad_xy[x,y]*diff_x+grad_y_squared[x,y]*diff_y)
                            elem_sum += diff_x*temp_1+diff_y*temp_2
                            
                    corner_sum.append((corn,elem_sum))        
                    
                    #temp_list= []
                    #temp_list.append((inv_corr[0,0]*temp_arr[0]+inv_corr[0,1]*temp_arr[1]))      
                    #temp_list.append((inv_corr[1,0]*temp_arr[0]+inv_corr[1,1]*temp_arr[1]))
                    
                    #refined_corner_list.append(temp_list)        
                if(len(corner_sum)>0):
                    corner_sum_sorted = sorted(corner_sum, key=lambda x:x[1])
                    refined_corner_list.append(corner_sum_sorted[0][0])
                for elem in window_corner:
                    corner_list.remove(elem)                
                    
        
        return refined_corner_list   
    
    
    def track_value(val):
        pass
        
        
    
    
    while(True):
        inp = input("Input 'c' to perform corner detection. Input 'h' for help. Input 'exit' to exit\n")
        inp = inp.strip()

        # Convert the image to grayscale using the openCV conversion function
        if(inp == "c"):
            image = cv2.imread(filename)
            image1 = cv2.imread(filename1)
            image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            image1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
 
            title_window = "Corner Detection"
            cv2.namedWindow(title_window)
            # create trackbars for controlling threshold
            cv2.createTrackbar('threshold',title_window,1,20,track_value)
            # create trackbars for controlling neighbour_size
            cv2.createTrackbar('neighbour',title_window,1,10,track_value)
            # create trackbars for controlling weight_trace
            cv2.createTrackbar('weight_k',title_window,1,500,track_value) 
            image1 = cv2.resize(image1, image.shape)           
            numpy_horizontal_concat = np.concatenate((image.copy(), image1.copy()), axis=1)            
            
            
            num_loop = 0
            while(True):
                num_loop += 1
                cv2.imshow(title_window,numpy_horizontal_concat)
                cv2.waitKey(0)
                if(num_loop == 5):
                    break
                            
                k = int(cv2.getTrackbarPos('weight_k',title_window))/1000
                threshold_value = int(cv2.getTrackbarPos('threshold',title_window))*1000000
                neighbour = int(cv2.getTrackbarPos('neighbour',title_window))
                window_size = 2*neighbour+1
                
                # Calling function to apply corner detection algorithm on image2
                corner_list1 = apply_harris(image.copy(), window_size, k, threshold_value)
                #print(corner_list1)
                
                # Calling function to apply corner detection algorithm on image3
                corner_list2 = apply_harris(image1.copy(), window_size, k, threshold_value)
                #print(corner_list2)
                
                image2 = cv2.cvtColor(image.copy(),cv2.COLOR_GRAY2BGR)
                image3 = cv2.cvtColor(image1.copy(),cv2.COLOR_GRAY2BGR) 
                
                # Matching feature points using feature vectors
                c1_match_c2 = {}
                c2_match_c1 = {}
                num = 1
                for corner1 in corner_list1:
                    distance = []
                    for corner2 in corner_list2:
                        distance.append([corner2,np.sqrt((corner1[0]-corner2[0])**2+(corner1[1]-corner2[1])**2)+(corner1[2]-corner2[2])**2])
                    # Sort in increasing order on the basis of distance
                    distance = sorted(distance, key=lambda x:x[1])
                    c1_match_c2[str(corner1)] = [distance[0][0],num]
                    c2_match_c1[str(distance[0][0])] = [corner1,num]
                    num += 1
                        
                        
                
                # Drawing rectangles om image2
                for corner in corner_list1:
                    x_loc = int(corner[0])
                    y_loc = int(corner[1])
                    # Draw a rectangle with red line borders of thickness of 2 px 
                    image2 = cv2.rectangle(image2, (x_loc-8,y_loc-8), (x_loc+12,y_loc+12), color=(0,0,255), thickness=1)
                    loc_num = str(c1_match_c2[str(corner)][1])
                    image2 = cv2.putText(image2,loc_num,(x_loc-10,y_loc-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255, 0, 0),1,cv2.LINE_AA )
                    
                # Drawing rectangles om image3
                for corner in corner_list2:
                    x_loc = int(corner[0])
                    y_loc = int(corner[1])
                    # Draw a rectangle with red line borders of thickness of 2 px 
                    image3 = cv2.rectangle(image3, (x_loc-8,y_loc-8), (x_loc+12,y_loc+12), color=(0,0,255), thickness=1)
                    
                    if(corner in list(np.array(list(c1_match_c2.values()))[:,0])):
                        loc_num = str(c2_match_c1[str(corner)][1])
                        image3 = cv2.putText(image3,loc_num,(x_loc-10,y_loc-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255, 0, 0),1,cv2.LINE_AA )                    
                numpy_horizontal_concat = np.concatenate((image2, image3), axis=1)        
            
            
            cv2.destroyAllWindows()
          
                  
        
        # Display a short description of the program, its command line arguments and the keys it supports .
        elif(inp == "h"):
            # Code to do
            print("\nPROGRAM DESCRIPTION : In this program, Corner detection is performed using Harris Corner Detection algorithm. ")
            print("It is totally implemented from scratch and corner localization is also done from scratch")
            print("\n COMMAND LINE ARGUMENT - (1). python main.py 'image_file1.jpg' 'image_file2.jpg'  (2). python main.py")
            print("\n SUPPORTED KEYS:-")
            print("(1). 'c' - Perform Corner Detection")
            print("(1). 'h' - Display a short description of the program, its command line arguments, and the keys it supports")
            print("(3). 'exit' - Exit the program\n")
            
        
        # Exit the program
        elif(inp == "exit"):
            break
    
        else:
            print(" Please input valid entry . For more info please input 'h' ")
        
        

            