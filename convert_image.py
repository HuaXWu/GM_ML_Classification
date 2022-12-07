from pyDeepInsight import ImageTransformer, LogScaler
import pandas as pd
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

"""
    convert matrix to image
    https://github.com/YDaiLab/PopPhy-CNN
"""
data = pd.read_excel("./data/T2D", header=None, index_col=0)
datainfo = data.T.fillna(0)
datainfo = datainfo.sort_values("label")
label = datainfo["label"]
positive = datainfo.iloc[217:, 1:]
negative = datainfo.iloc[0:217, 1:]
ln = LogScaler()
positive_norm = ln.fit_transform(positive)
negative_norm = ln.transform(negative)

positive_norm = positive_norm.fillna(0)
negative_norm = negative_norm.fillna(0)
it = ImageTransformer(feature_extractor='tsne',
                      pixels=50, random_state=1701,
                      n_jobs=-1)
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, perplexity=30, metric='cosine',
            random_state=1701, n_jobs=-1)

it = ImageTransformer(feature_extractor=tsne, pixels=50)
plt.figure(figsize=(10, 10))
it.fit(positive_norm, plot=True)
fdm = it.feature_density_matrix()
fdm[fdm == 0] = np.nan

plt.figure(figsize=(5, 5))

ax = sns.heatmap(fdm, cmap="viridis", linewidths=0.01,
                 linecolor="lightgrey", square=True)
ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
for _, spine in ax.spines.items():
    spine.set_visible(True)
_ = plt.title("Genes per pixel")
px_sizes = [25, (25, 50), 50, 100]

fig, ax = plt.subplots(1, len(px_sizes), figsize=(25, 7))
for ix, px in enumerate(px_sizes):
    it.pixels = px
    fdm = it.feature_density_matrix()
    fdm[fdm == 0] = np.nan
    cax = sns.heatmap(fdm, cmap="viridis", linewidth=0.01,
                      linecolor="lightgrey", square=True,
                      ax=ax[ix], cbar=False)
    cax.set_title('Dim {} x {}'.format(*it.pixels))
    for _, spine in cax.spines.items():
        spine.set_visible(True)
    cax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    cax.yaxis.set_major_locator(ticker.MultipleLocator(5))
plt.tight_layout()

it.pixels = 25
positive_img = it.fit_transform(positive_norm)
negative_img = it.transform(negative_norm)
for i in range(168, positive_img.shape[0]):
    img = Image.fromarray(np.uint8(positive_img[i] * 255))  # 将数组转化回图
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)
    plt.savefig("./data/T2D/images/positive/sample{}.png".format(i + 1), dpi=300, bbox_inches='tight', pad_inches=-0.1)
