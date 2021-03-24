# import numpy as np
# import matplotlib.pyplot as plt

# # Fixing random state for reproducibility
# np.random.seed(19680801)

# # the random data
# x = np.random.randn(1000)
# y = np.random.randn(1000)

# # definitions for the axes
# left, width = 0.1, 0.65
# bottom, height = 0.1, 0.65
# spacing = 0.005


# rect_scatter = [left, bottom, width, height]
# rect_histx = [left, bottom + height + spacing, width, 0.2]
# rect_histy = [left + width + spacing, bottom, 0.2, height]

# # start with a rectangular Figure
# plt.figure(figsize=(8, 8))

# ax_scatter = plt.axes(rect_scatter)
# ax_scatter.tick_params(direction='in', top=True, right=True)
# ax_histx = plt.axes(rect_histx)
# ax_histx.tick_params(direction='in', labelbottom=False)
# ax_histy = plt.axes(rect_histy)
# ax_histy.tick_params(direction='in', labelleft=False)

# # the scatter plot:
# ax_scatter.scatter(x, y)

# # now determine nice limits by hand:
# binwidth = 0.25
# lim = np.ceil(np.abs([x, y]).max() / binwidth) * binwidth
# ax_scatter.set_xlim((-lim, lim))
# ax_scatter.set_ylim((-lim, lim))

# bins = np.arange(-lim, lim + binwidth, binwidth)
# ax_histx.hist(x, bins=bins)
# ax_histy.hist(y, bins=bins, orientation='horizontal')

# ax_histx.set_xlim(ax_scatter.get_xlim())
# ax_histy.set_ylim(ax_scatter.get_ylim())

# plt.show()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy import stats
import pandas as pd

label = ['Setosa','Versicolor','Virginica'] # List of labels for categories
cl = ['b','r','y'] # List of colours for categories
sample_size = 20 # Number of samples in each category

dir_opt = '/datainfo'
path = '.' + dir_opt + '/result/5-fold-flabel01/epoch_99'
pred_dl_input_df = pd.read_csv(path + '/PredTrainingInput_flabel.txt', delimiter = ',')
print(pred_dl_input_df.corr(method = 'pearson'))
# CALCULATE THE MSE FOR TRAIN
train_auc_list = list(pred_dl_input_df['AUC'])
train_auc = np.array(train_auc_list)
train_pred_list = list(pred_dl_input_df['Pred Score'])
train_pred = np.array(train_pred_list)
train_false_list = list(pred_dl_input_df['False AUC'])
train_false = np.array(train_false_list)

# Generate random data for each categorical variable:
x = train_auc
y = train_pred

# Set up 4 subplots as axis objects using GridSpec:
gs = gridspec.GridSpec(2, 2, width_ratios=[1,3], height_ratios=[3,1])
# Add space between scatter plot and KDE plots to accommodate axis labels:
gs.update(hspace=0.3, wspace=0.3)

# Set background canvas colour to White instead of grey default
fig = plt.figure()
fig.patch.set_facecolor('white')

ax = plt.subplot(gs[0,1]) # Instantiate scatter plot area and axis range
ax.set_xlim(x.min(), x.max())
ax.set_ylim(y.min(), y.max())
ax.set_xlabel('AUC')
ax.set_ylabel('Pred')

axl = plt.subplot(gs[0,0], sharey=ax) # Instantiate left KDE plot area
axl.get_xaxis().set_visible(False) # Hide tick marks and spines
axl.get_yaxis().set_visible(False)
axl.spines["right"].set_visible(False)
axl.spines["top"].set_visible(False)
axl.spines["bottom"].set_visible(False)

axb = plt.subplot(gs[1,1], sharex=ax) # Instantiate bottom KDE plot area
axb.get_xaxis().set_visible(False) # Hide tick marks and spines
axb.get_yaxis().set_visible(False)
axb.spines["right"].set_visible(False)
axb.spines["top"].set_visible(False)
axb.spines["left"].set_visible(False)

axc = plt.subplot(gs[1,0]) # Instantiate legend plot area
axc.axis('off') # Hide tick marks and spines


ax.scatter(x, y)

kde = stats.gaussian_kde(x)
xx = np.linspace(x.min(), x.max(), 1000)
axb.plot(xx, kde(xx), color='black')

kde = stats.gaussian_kde(y)
yy = np.linspace(y.min(), y.max(), 1000)
axl.plot(kde(yy), yy, color='black')


plt.show()