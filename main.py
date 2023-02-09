from solver import EdgeSimilaritySolver
import model as network
import cv2, torch
import numpy as np

class SegNet():
    

    def __init__(self, checkpoint_path, img_w, img_h):

        self.model = network.deeplabv3plus_resnet101(num_classes=1, output_stride=16)
        network.convert_to_separable_conv(self.model.classifier)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval() 

        self.img_w = img_w
        self.img_h = img_h
  

    def pre_process(self, img):
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)
        img = img.transpose((2, 0, 1))
        if img.max() > 1:
            img = img / 255    
        return img

    def predict_img(self, img):

        img = cv2.resize(img, (self.img_w, self.img_h))
        img = torch.from_numpy(self.pre_process(np.array(img)))

        img = img.unsqueeze(0)
        img = img.to(device=self.device, dtype=torch.float32)

        with torch.no_grad():
            output = self.model(img)
            probs = torch.sigmoid(output)
            probs = probs.squeeze(0).cpu()
            full_mask = probs.squeeze().cpu().numpy()

        result = full_mask / np.max(full_mask)
        result = (result * 255).astype(np.uint8)
        return result




if __name__ == "__main__":
    
    IMG_PATH = "data/test_imgs/c1_0.png"
    CHECKPOINT_SEG = "checkpoints/CP_segmentation.pth"

    # params
    N_KNN = 8
    TH_EDGES_SIMILARITY = 0.1
    SAMPLING_RATIO = 0.15
    TH_MASK = 127
    IMG_W = 640
    IMG_H = 360

    solver = EdgeSimilaritySolver(n_knn=N_KNN, th_edges_similarity=TH_EDGES_SIMILARITY, sampling_ratio=SAMPLING_RATIO, th_mask=TH_MASK)
    network_seg = SegNet(checkpoint_path=CHECKPOINT_SEG, img_w=IMG_W, img_h=IMG_H)
    
    # get img to test
    img = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (640, 360))

    # get mask
    mask_img = network_seg.predict_img(img)

    # solver
    paths, mask_output, points_new = solver.exec(mask_img, img)

    cv2.imshow("output", mask_output)
    cv2.waitKey(0)

    
