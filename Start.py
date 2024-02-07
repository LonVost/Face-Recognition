from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt

Mojito = False

class Security:
    def self(self):
        print("")
    def takePic(self):	#anlık olarak kameradan fotoğraf çekip kayıt edelim
        camera = cv2.VideoCapture(0)
        return_value, image = camera.read()
        flipped_image = cv2.flip(image, 1)
        cv2.imwrite('...\\test.jpg', flipped_image)	#kayıt yerini belirtelim
        camera.release()
        cv2.destroyAllWindows()

    def controlPic(self):
        img1_path = "...\\face_db\\..."		#kontrol edeceğimiz resimin yoluun ve adını belirtelim
        img2_path = "...\\test.jpg"		#aldığımız anlık resmin yolunu ve adını belirtelim

        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        plt.imshow(img1[:,:,::-1])
        plt.show()
        plt.imshow(img2[:,:,::-1])
        plt.show()

        resp = DeepFace.verify(img1_path, img2_path, model_name="DeepFace")
        Mojito = resp["verified"]
        return Mojito  # return ile Mojito değerini döndürelim

security_obj = Security()
security_obj.takePic()
result = security_obj.controlPic()
print(result)  # yüz tanıma sonucunu yazdıralım
