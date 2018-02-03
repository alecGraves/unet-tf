import os
import cv2
import numpy as np
import csv
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import euclidean_distances

delim = ','

with open('eggs.csv', 'w') as csvfile:
    csvfile.write('ImageId' + delim + 'WKT_Pix' + '\n')

filenames = ['C:\\Users\\Alec\\Desktop\\test_masks\\AOI_2_Vegas_Roads_Test_Public\\masks\\RGB-PanSharpen_AOI_2_Vegas_img9.jpeg']
for filename in filenames:
    img = cv2.imread(filename, 0)
    ret,img = cv2.threshold(img,50,255,0)
    area = np.sum(img == 255)

    def average_line(line, image):
        x1, y1, x2, y2 = line
        dx = x2 - x1
        dy = y2 - y1
        vals = []
        if dx != 0:
            if dx  <0:
                direction = -1
            else:
                direction = 1
            for x in range(x1, x2, direction):
                y = y1 + dy * (x - x1) // dx
                vals.append(image[x, y])
        else:
            if dy  <0:
                direction = -1
            else:
                direction = 1
            for y in range(y1, y2, direction):
                x = x1 + dx * (y - y1) // dy
                vals.append(image[x, y])
        return np.mean(vals)

    points = np.argwhere(img == 255)
    lines = []
    density_k = int(300/163125*area)
    if density_k > 1:
        kmeans = MiniBatchKMeans(n_clusters=density_k, random_state=0, max_iter=3).fit(points)
        best_points = kmeans.cluster_centers_.astype(np.int64)

        print('cluster complete')
        dist = euclidean_distances(best_points)
        print(dist.shape)
    
        for i in range(10, best_points.shape[0]):
            closest = dist[i].argsort()[1:5]
            for j in range(len(closest)):
                if closest[j] < 20:
                    continue
                line = np.concatenate((best_points[i], best_points[closest[j]]), axis=-1)
                density = average_line(line, img)
                y1, x1, y2, x2 = line
                if density > 200:
                    if (x1, y1) < (x2, y2):
                        lines.append([x1, y1, x2, y2])
                    else:
                        lines.append([x2, y2, x1, y1])
                # Display
                # line_image = img//3
                # line_img = cv2.line(line_image,(x1,y1),(x2,y2),(255),1)
                # cv2.imshow("cont",line_img)
                # cv2.waitKey(10)
        print(len(lines))
        lines = set([tuple(line) for line in lines])
        print(len(lines))
        # gmm = GaussianMixture(n_components=50).fit(points)
        # best_points = gmm.means_.astype(np.uint32)

        img = img//3
        for point in best_points:
            img = cv2.circle(img, tuple(point[::-1]), 3, 255, 1)
        for x1,y1,x2,y2 in lines:
            cv2.line(img,(x1,y1),(x2,y2),(255),1)

        cv2.imshow("cont",img)
        cv2.waitKey(10)
        print(best_points.shape)

    csv_dataname = os.path.basename(filename)
    print(csv_dataname)
    csv_dataname = csv_dataname[15:csv_dataname.find('.')]
    print(csv_dataname)

    if len(lines) < 1:
        data = 'LINESTRING EMPTY'
    else:
        data = '"LINESTRING ('
        for i, line in enumerate(lines):
            if i > 0:
                data = data + ', ' + str(line[0]) + ' ' + str(line[1]) + ', ' + str(line[2]) + ' ' + str(line[3])
            else:
                data = data + str(line[0]) + ' ' + str(line[1]) + ', ' + str(line[2]) + ' ' + str(line[3])
        data = data + ')"'

    with open('eggs.csv', 'a') as csvfile:
        csvfile.write(csv_dataname + delim + data + '\n')