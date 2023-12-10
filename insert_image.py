import cv2

# display resolution: FHD
target_width = 1920	
target_height = 1080

image_name = input('enter your image name: ')

image_path = './image/'+image_name+'.jpg'

image = cv2.imread(image_path)

if image is None:
	print("\nEroor: File not found or unable to load.")
	exit()

original_height, original_width = image.shape[:2]

resized_image = cv2.resize(image, (target_width, target_height))

cv2.imshow("Resized Image", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()