from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import networkx as nx
import os


def load_png(png_filename):
    png = Image.open(png_filename)
    img = np.array(png)
    return img


def get_circles(img):
    gray = img.copy()
    circles = cv2.HoughCircles(gray.astype('uint8'), cv2.HOUGH_GRADIENT,1,20, minRadius=2, maxRadius=4, param1=20, param2=5)
    if circles is None:
        print("Error: detected no circles...")
        exit()
    if circles is not None and len(circles) != 2:
        print("Error: detected more or less than 2 circles...")
        output = img.copy()
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(output, (x, y), r, 255, 2)

        new = np.hstack([img, output])
        plt.imshow(new)
        exit()
    return circles


def do_dilation(img):
    kernel = np.ones((2, 2), np.uint8)
    img_dilation = cv2.dilate(img.astype('uint8'), kernel, iterations=1)
    img_dilation[img_dilation > 0] = 1
    return img_dilation


def construct_nodes(image):
    # adapted from: https://stackoverflow.com/questions/63653267/how-to-create-a-graph-with-an-images-pixel
    #CONSTRUCTION OF HORIZONTAL EDGES
    hx, hy = np.where(image[1:] & image[:-1]) #horizontal edge start positions
    h_units = np.array([hx, hy]).T
    h_starts = [tuple(n) for n in h_units]
    h_ends = [tuple(n) for n in h_units + (1, 0)] #end positions = start positions shifted by vector (1,0)
    horizontal_edges = zip(h_starts, h_ends)

    #CONSTRUCTION OF VERTICAL EDGES
    vx, vy = np.where(image[:,1:] & image[:,:-1]) #vertical edge start positions
    v_units = np.array([vx, vy]).T
    v_starts = [tuple(n) for n in v_units]
    v_ends = [tuple(n) for n in v_units + (0, 1)] #end positions = start positions shifted by vector (0,1)
    vertical_edges = zip(v_starts, v_ends)

    return horizontal_edges, vertical_edges

 
def construct_graph(img_dilation):
    G = nx.Graph()
    horizontal_edges, vertical_edges = construct_nodes(img_dilation)
    G.add_edges_from(horizontal_edges)
    G.add_edges_from(vertical_edges)

    # get diagonals
    nodes = list(G.nodes)
    for n in nodes:
        upR = (n[0]+1, n[1]+1)
        downR = (n[0]-1, n[1]+1)
        upL = (n[0]+1, n[1]-1)
        downL = (n[0]-1, n[1]-1)
        if upR in nodes:
            G.add_edge(n,upR)
        if downR in nodes:
            G.add_edge(n,downR)
        if upL in nodes:
            G.add_edge(n,upL)
        if downL in nodes:
            G.add_edge(n,downL)
    return G


def pad_path(path):
    padded_path = []
    for p in path:
        padded_path.append((p[0]+1, p[1]+1)) # downR
        padded_path.append((p[0]+1, p[1])) # down
        padded_path.append((p[0]+1, p[1]-1)) # downL
        padded_path.append((p[0], p[1]+1)) # R
        padded_path.append((p[0], p[1]-1)) # L
        padded_path.append((p[0]-1, p[1]+1)) # upR
        padded_path.append((p[0]-1, p[1])) # up
        padded_path.append((p[0]-1, p[1]-1)) # upL
    return padded_path


def draw_path(G, circles):
    adj = [(n, nbrdict) for n, nbrdict in G.adjacency()]
    adj_dict = {}
    for el in adj:
        adj_dict[el[0]] = [key for key in el[1].keys()]

    num_circles = len(circles)
    anchors = []
    for i in range(num_circles):
        if circles[i][2] > 5:
            print("have big radii detected, may need to fix hough")
        anchors.append((circles[i][1], circles[i][0])) # swap coords

    path = nx.dijkstra_path(G, anchors[0], anchors[1])
    padded_path = pad_path(path)
    return padded_path


def create_mask(img, padded_path):
    path_img = np.zeros_like(img)
    path_img[tuple(np.array(padded_path).T)] = 1
    return path_img


def process_directory(dir, save_dir):
    for png_filename in os.listdir(dir):
        img = load_png(png_filename)
        img_dilation = do_dilation(img)
        G = construct_graph(img_dilation)
        circles = get_circles(img)
        padded_path = draw_path(G, circles)
        mask = create_mask(img, padded_path)
        np.save(mask, save_dir)