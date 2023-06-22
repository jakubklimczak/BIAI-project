import cv2

# Try accessing cameras from index 0 to 9
'''
for i in range(10):
    camera = cv2.VideoCapture(i)
    if camera.isOpened():
        print(f"Camera found at index {i}")
        camera.release()
'''

camera = cv2.VideoCapture(0)
ret, frame = camera.read()

if ret:
    cv2.imwrite("photo.jpg", frame)
    print("Photo saved successfully!")
else:
    print("Failed to capture photo.")

camera.release()
