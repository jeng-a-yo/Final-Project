import cv2
import os

for i in os.listdir("archive-3/extracted_images"):
    if i != ".DS_Store":
        for j in os.listdir(f"archive-3/extracted_images/{i}"):
            if True:
                image = cv2.imread(f"archive-3/extracted_images/{i}/{j}")
                invertImage = cv2.bitwise_not(image)
                cv2.imwrite(f"archive-3/extracted_images/{i}/{j}", invertImage)

# for j in os.listdir(f"bhmsds/symbols"):
#     if True:
#         image = cv2.imread(f"bhmsds/symbols/{j}")
#         invertImage = cv2.bitwise_not(image)
#         cv2.imwrite(f"bhmsds/symbols/{j}", invertImage)

# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# inverted_image = cv2.bitwise_not(gray_image)

# cv2.imshow('Original Image', gray_image)
# cv2.imshow('Inverted Image', inverted_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


