# DTNet_Last
This is the code for my paper.
Dual Triplet Network for image Zero-shot Learning(https://www.sciencedirect.com/science/article/pii/S092523121931330X)
If you can't download the paper, you can contact me.
Email: wang1104014663@gmail.com

![1-s2 0-S092523121931330X-gr1](https://user-images.githubusercontent.com/18210788/111248788-8eb4a180-8645-11eb-8189-bad67968b434.jpg)

Fig. 1. Diagram for the proposed framework. First, the DTNet maps the attribute features into the visual space with the mapping network. Then, it employs two triplet networks for learning the visual-semantic alignment.

![image](https://user-images.githubusercontent.com/18210788/111248818-9e33ea80-8645-11eb-8fa1-809e507fc573.png)

Fig. 2. Illustration for the proposed DTNet. It shows in detail the composition and input of AOTN and VOTN. Note the two metric networks share parameters.

![image](https://user-images.githubusercontent.com/18210788/111248834-a7bd5280-8645-11eb-9faf-fe232da370a4.png)

Fig. 3. The illustration of and DTNet-WHTM and DTNet. Different colors represent different categories. The pentagram denotes the attribute features, and the triangle denotes the visual features. Different modalities with the same category are forced to be close, while those from different categories are forced to keep away.
