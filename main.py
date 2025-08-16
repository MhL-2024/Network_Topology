
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticks
from matplotlib import cm
from matplotlib.colors import ListedColormap
from scipy import stats
import seaborn as sns
import pingouin as pg


plt.rc('font', family='Arial')


CONFIG = {
    'font_size': 33,
    'scatter_color': '#E07B54',
    'scatter_size': 230,
    'box_width': 3,
    'cbar_fraction': 0.04,
    'cbar_pad': 0.01
}

def significance_stars(p):
    """Return significance stars for p-values."""
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return ''



def bingenerator(df, field, bin_num):
    """
    Generate bins for a given field based on 1st and 99th percentile.
    """
    low = df[field].quantile(0.01)
    high = df[field].quantile(0.99)
    interval = (high - low) / (2 * bin_num - 1)
    bins = np.arange(low, high, interval).tolist()
    bins = [df[field].min()] + bins + [df[field].max()]
    return bins



def plot_map(ax, legend_label, gdf, field, cmap_name, vmax, vmin, label):
    
    size0 = CONFIG['font_size']
    plt.rc('font', family='Arial', size=size0)


    # Color map
    base_cmap = cm.get_cmap(cmap_name, 512)
    cmap = ListedColormap(base_cmap(np.linspace(0, 1, 256)))

    gdf.plot(ax=ax, column=field, cmap=cmap, vmax=vmax, vmin=vmin,
             edgecolor="black", linewidth=0.1, zorder=2)

 
    cbar = plt.colorbar(ax.get_children()[0], ax=ax, location='top',
                        fraction=CONFIG['cbar_fraction'], pad=CONFIG['cbar_pad'], shrink=0.5, extend='both')
    cbar.ax.tick_params(length=8, pad=0.08, width=2, labelsize=size0)
    cbar.set_label(label, labelpad=13, fontsize=size0)
    cbar.outline.set_linewidth(2.5)

    # Custom ticks
    tick_dict = {
        'Tk_c': [1.7, 2.1, 2.5],
        'logAI': [-1, -0.5, 0],
        'SR': [0.4, 0.5, 0.6],
        'Slope': [0.03, 0.1, 0.17]
    }
    if field in tick_dict:
        cbar.set_ticks(tick_dict[field])

    # Axis and legend
    ax.axis('off')
    ax.text(0.07, 1.22, legend_label, transform=ax.transAxes,
            fontsize=size0 + 8, fontweight='bold', ha='left', va='top', color='black')


def bindata_scatter_plot(ax1,fig_legend,df,classname,yname,bin_values,maxval,minval):

    bins = pd.cut(df[classname], bins=bin_values)
    df[classname+'_bins'] = bins
    
    # Group by the bins and calculate the mean for each group
    dftmp = df.groupby(classname+'_bins').agg({classname: ['mean','count','std'], yname: ['mean','std']}).reset_index()
    
    dfplot = pd.DataFrame({
        classname: dftmp[classname]['mean'],
        yname: dftmp[yname]['mean'],
        yname + '_std': dftmp[yname]['std'],
        'count': dftmp[classname]['count']
    })
    
    yerrname=yname+'_err'
    dfplot[yerrname]=dftmp[yname]['std']/np.sqrt(dftmp[classname]['count'])
    
    xerrname = classname+'_err'
    dfplot[xerrname]=dftmp[classname]['std']/np.sqrt(dftmp[classname]['count'])
    
    size0=33 
    dotColor = '#E07B54'
    s_size = 230
    
    scatter_a = ax1.scatter(x=dfplot[classname], y=dfplot[yname], c=dotColor, s=s_size, edgecolors='black',alpha=1)
    dfplot.plot(classname,yname,xerr=xerrname,yerr=yerrname, alpha=1,ax=ax1,ls='none',label=None,legend=False,zorder=0,ecolor='#999999',elinewidth=3,capsize=5)
    
    ax1.set_ylim(1.6, 2.8)
    if classname=='logAI':
       ax1.set_xlabel('log$_{10}$[AI]')   
       ax1.set_ylabel('Tokunaga \nparameter c', fontsize=size0, labelpad=4) 
    
         
    elif classname=='Slope':
       ax1.set_xlabel('Channel slope')
       ax1.tick_params(axis='y', which='both', left=False, labelleft=False)
       
    elif classname=='SR':
       ax1.set_xticks([0.3,0.55,0.8])
       ax1.set_xlabel('Slope ratio')
       ax1.tick_params(axis='y', which='both', left=False, labelleft=False)
    
    # ---- spearman correlation ------
    df_clean = df[[classname, yname]].dropna()
    spearman_corr, p_value = stats.spearmanr(df_clean[classname], df_clean[yname])
    
    rho_stars = significance_stars(p_value)
    rho_text = f"ρ={spearman_corr:.2f}"
    
    t_rho = ax1.text(x=0.23, y=0.12, s=rho_text,
                         fontsize=size0, color='black',horizontalalignment='center', verticalalignment='center',
                         transform=ax1.transAxes)
    
    fig = ax1.get_figure()
    fig.canvas.draw()
    bbox = t_rho.get_window_extent(renderer=fig.canvas.get_renderer())

    inv = ax1.transAxes.inverted()
    bbox_axes = inv.transform([[bbox.x1, bbox.y1]])
    star_x = bbox_axes[0][0] - 0.02  # Slightly to the right
    star_y = 0.13  # Slightly above

    ax1.text(x=star_x, y=star_y, s=rho_stars,
         fontsize=size0-3, fontweight='bold',
         color='black', transform=ax1.transAxes)
       
    
    boxwidth=3
    ax1.tick_params(labelsize=size0,direction='in',width=1,length=8,pad=10)
    ax1.tick_params(which='minor',direction='in')
    for spine in ['bottom', 'left', 'right', 'top']:
        ax1.spines[spine].set_linewidth(boxwidth)
    
    ax1.text(0.01, 1.2, fig_legend, transform=ax1.transAxes, fontsize=size0+8, fontweight='bold',ha='left', va='top', 
                 color='black')
    
    return dfplot

def classify_quantile(df, col, labels=None):
    """divide dataset"""
    q20, q50 = df[col].quantile([0.2, 0.5])
    if labels is None:
        labels = [
            f'{col} ≤ {q20:.2f}',
            f'{q20:.2f} < {col} ≤ {q50:.2f}',
            f'{col} > {q50:.2f}'
        ]
    df[f'{col}_class'] = pd.cut(
        df[col],
        bins=[-np.inf, q20, q50, np.inf],
        labels=labels
    )
    return df

def bin_stats(df, angle_col, bin_col='Tk_bin'):
    """calculate stats for each group"""
    return df.groupby(bin_col).agg(
        Tk_c_mean=('Tk_c', 'mean'),
        angle_mean=(angle_col, 'mean'),
        angle_sem=(angle_col, lambda x: x.std(ddof=1) / np.sqrt(len(x)))
    ).reset_index()


        
def custom_bin_and_plot(df, class_col,x_range,y_range, ax,palette):
    markers = ['o', 'D', '^'] 
    marksize=[20,18,21]
    for i, (cls, group) in enumerate(df.groupby(class_col)):
        group = group.copy()

        # Get quantiles
        q01 = group['Tk_c'].quantile(0.01)
        q99 = group['Tk_c'].quantile(0.99)

        # 18 equal-width bins between q01 and q99
        mid_bins = np.linspace(q01, q99, 9)  # 10 intervals = 11 edges
        bin_edges = np.concatenate(([-np.inf], [q01], mid_bins[1:-1], [q99], [np.inf]))

        # Assign bins
        group['Tk_c_bin'] = pd.cut(group['Tk_c'], bins=bin_edges, labels=False)

         # Group by bins and compute mean and SEM
        binned = group.groupby('Tk_c_bin').agg(
            Tk_c_mean=('Tk_c', 'mean'),
            angle_mean=('angle', 'mean'),
            angle_sem=('angle', lambda x: x.std(ddof=1) / np.sqrt(len(x)))
        ).reset_index()

        # Plot with error bars
        ax.errorbar(binned['Tk_c_mean'], binned['angle_mean'], yerr=binned['angle_sem'],
                    fmt=markers[i], markersize=marksize[i], capsize=5,ecolor='gray',markeredgecolor='black', 
                    label=f'{cls}', color=palette[i])
        
       
        boxwidth=3
        ax.tick_params(labelsize=36,direction='in',width=1,length=8,pad=10)
        ax.tick_params(which='minor',direction='in')
        ax.spines['bottom'].set_linewidth(boxwidth)
        ax.spines['left'].set_linewidth(boxwidth)
        ax.spines['right'].set_linewidth(boxwidth)
        ax.spines['top'].set_linewidth(boxwidth)
        
        ax.set_ylim(y_range)
        ax.set_xlim(x_range)


def ecdfplot(dfplot, classname, num_classes, bin_values, plotfield):
    dfplot=dfplot.rename(columns={'Tk_c':'c'})
    if classname == 'Tk_c':
        classname='c'
    
    dfplot[classname+'_class'] = pd.cut(dfplot[classname], bins=bin_values)
    fig = plt.figure(figsize=(8, 6))
    font = {'family': 'Arial', 'size': 20}
    plt.rc('font', size=20)
    ax = fig.add_subplot(111)
    
    crest_palette = sns.color_palette("rocket_r", n_colors=num_classes)
    start_color = 0  # Start color index
    end_color = 1  # End color index (adjust as needed)

    class_palette = sns.color_palette(crest_palette.as_hex()[int(start_color * num_classes):int(end_color * num_classes)], n_colors=num_classes)
    
    ecdf_plot = sns.ecdfplot(data=dfplot, x=plotfield,ax=ax,hue=classname+'_class', linewidth=3,palette=class_palette,zorder=1)
    
    # Plot mean points
    class_means = dfplot.groupby(classname+'_class')[plotfield].mean().reset_index()
    ecdf_y=[]
    
    for x in class_means[plotfield]:
        dftmp = dfplot[dfplot[classname+'_class']==class_means[class_means[plotfield]==x][classname+'_class'].iloc[0]]
        ytmp=dftmp[dftmp[plotfield]<=x].shape[0]/dftmp.shape[0]
        ecdf_y.append(ytmp)
    class_means['ecdf']=ecdf_y
    class_means.plot.scatter(plotfield, 'ecdf', marker='o',c=class_palette,edgecolors='black', s=150,ax=ax,zorder=2)
    
    
    # Set labels and title
    if plotfield=='Bifurcation':
        plotfield='Bifurcation angle[°]'
    elif plotfield=='Sidebranch':
        plotfield='Side-branching angle[°]'
        ax.legend_.remove()
      
    ax.tick_params(labelsize=20,direction='in',width=1,length=8,pad=10)
    ax.tick_params(which='minor',direction='in')
    for spine in ['bottom', 'left', 'right', 'top']:
        ax.spines[spine].set_linewidth(3)
        
    ax.yaxis.set_major_formatter(mticks.FormatStrFormatter('%.2f'))
    
    ax.set_xlabel(plotfield, fontsize=26)
    ax.set_ylabel('Proportion', fontsize=26)
    ax.tick_params(labelsize=26)
    

# %%

# ------------------------------
# Main plotting
# ------------------------------
def main():
    # Load data
    hex10000 = gpd.read_file(r'U:\fromHardDisk\Tokunaga\Manuscript\NC\submission\dataset_submission\hexagon10000.shp')
    Tk = pd.read_csv(r'U:\fromHardDisk\Tokunaga\Manuscript\NC\submission\dataset_submission\Networks_5th_order.csv')
    Tk_angles = pd.read_csv(r'U:\fromHardDisk\Tokunaga\Manuscript\NC\submission\dataset_submission\Classified_angles_vs_Tk_c.csv')
    grouped_by_id0=pd.read_csv(r'U:\fromHardDisk\Tokunaga\Manuscript\NC\submission\dataset_submission\basin_aveDeltaHS_junction.csv')

    # ************* figure 2 *************************************
    fig = plt.figure(figsize=(18, 18))
    gs = gridspec.GridSpec(4, 10, figure=fig, wspace=0.01, hspace=0.11,
                           height_ratios=[1, 0.09, 1, 0.65],
                           width_ratios=[0.9, 1, 1, 0.08, 1, 1, 0.08, 1, 1, 0.3])

    # Generate bins
    SR_bin50 = bingenerator(Tk,'SR',10)
    logAI_bin50 = bingenerator(Tk,'logAI',10)
    Slope_bin50 = bingenerator(Tk,'Slope',10)

    # Map plotting
    fields = [
        ('a', 'Tk_c', 'viridis', 0, 0, 0.90, 0.02, 'Tokunaga parameter c'),
        ('b', 'logAI', 'viridis', 0, 5, 0.95, 0.01, 'log$_{10}$[AI]'),
        ('c', 'SR', 'viridis_r', 2, 0, 0.96, 0.05, 'Slope ratio'),
        ('d', 'Slope', 'viridis_r', 2, 5, 0.95, 0.20, 'Channel slope')
    ]
    
    for fig_lab, field, cmap, row, col, vmax_q, vmin_q, label in fields:
        ax = fig.add_subplot(gs[row, col:col+5])
        vmin_val = hex10000[field].quantile(vmin_q)
        vmax_val = hex10000[field].quantile(vmax_q)
        plot_map(ax, fig_lab, hex10000, field, cmap, vmax_val, vmin_val, label)

    maxval = 0.6
    minval = 0.15
    ax1 = fig.add_subplot(gs[3, 1:3])
    bindata_scatter_plot(ax1,'e',Tk,'logAI','Tk_c',logAI_bin50,maxval,minval)

    ax2 = fig.add_subplot(gs[3, 4:6])
    bindata_scatter_plot(ax2,'f',Tk,'SR','Tk_c',SR_bin50,maxval,minval)

    ax3 = fig.add_subplot(gs[3, 7:9])
    bindata_scatter_plot(ax3,'g',Tk,'Slope','Tk_c',Slope_bin50,maxval,minval)
    
    # ****************** figure 3 ********************************
    dfplot1 = Tk_angles[Tk_angles['angle_type'] == 'sidebranch'].copy()
    dfplot1 = classify_quantile(dfplot1, 'meanAI',
                                labels=[f'AI ≤ {dfplot1["meanAI"].quantile(0.2):.2f}',
                                        f'{dfplot1["meanAI"].quantile(0.2):.2f} < AI ≤ {dfplot1["meanAI"].quantile(0.5):.2f}',
                                        f'AI > {dfplot1["meanAI"].quantile(0.5):.2f}'])
    dfplot1 = classify_quantile(dfplot1, 'meanSlope',
                                labels=[f'S ≤ {dfplot1["meanSlope"].quantile(0.2):.2f}',
                                        f'{dfplot1["meanSlope"].quantile(0.2):.2f} < S ≤ {dfplot1["meanSlope"].quantile(0.5):.2f}',
                                        f'S > {dfplot1["meanSlope"].quantile(0.5):.2f}'])

    dfplot2 = Tk_angles[Tk_angles['angle_type'] == 'bifurcation'].copy()
    dfplot2 = classify_quantile(dfplot2, 'meanAI', labels=['Low', 'Medium', 'High'])
    dfplot2 = classify_quantile(dfplot2, 'meanSlope', labels=['Low', 'Medium', 'High'])

    q01, q99 = Tk['Tk_c'].quantile([0.01, 0.99])
    mid_bins = np.linspace(q01, q99, 9)
    bin_edges = np.concatenate(([-np.inf], [q01], mid_bins[1:-1], [q99], [np.inf]))
    Tk['Tk_bin'] = pd.cut(Tk['Tk_c'], bins=bin_edges, labels=False)


    fig = plt.figure(figsize=(16, 18))
    gs = gridspec.GridSpec(5, 2, height_ratios=[1, 0.22, 1, 1, 0.1], hspace=0.1, wspace=0.05)
    
    rows = [0, 2, 3]
    axes = [fig.add_subplot(gs[r, c]) for r in rows for c in range(2)]
    axes = np.array(axes).reshape(3, 2)

    size0 = 36
    boxwidth=3

    palette1 = sns.color_palette("YlOrBr_r", n_colors=3)
    palette2 = sns.color_palette("BuGn", n_colors=3)
    palette3 = sns.color_palette("Purples", n_colors=3)

    # boxplot
    sns.boxplot(data=grouped_by_id0, x='deltaHS', y='mean_SR', color='#03fce8', 
                width=0.3,
                showfliers=False,
                linewidth=3,
                ax=axes[0][0])
    axes[0][0].set_xlabel('ΔHS')
    axes[0][0].set_ylabel('Basin-averaged\nSR')
    # ====== add overall side-alpha, bifurcation in the first row =====

    dfplot2['Tk_bin'] = pd.cut(dfplot2['Tk_c'], bins=bin_edges, labels=False)
    dfplot1['Tk_bin'] = pd.cut(dfplot1['Tk_c'], bins=bin_edges, labels=False)

    binned_big = bin_stats(dfplot2, 'angle')
    binned_small = bin_stats(dfplot1, 'angle')
    overall_angle = bin_stats(Tk, 'Branching')


    axes[0][1].errorbar(overall_angle['Tk_c_mean'], overall_angle['angle_mean'], yerr=overall_angle['angle_sem'],
                fmt='^',markersize=21, color=palette3[0],markeredgecolor='black',  label='Branching', capsize=4)
    axes[0][1].errorbar(binned_big['Tk_c_mean'], binned_big['angle_mean'], yerr=binned_big['angle_sem'],
                fmt='o',markersize=20, color=palette3[1],markeredgecolor='black',  label='Bifurcation', capsize=4)
    axes[0][1].errorbar(binned_small['Tk_c_mean'], binned_small['angle_mean'], yerr=binned_small['angle_sem'],
                fmt='D',markersize=18, color=palette3[2],markeredgecolor='black',  label='Side-branching', capsize=4)

    axes[0][1].set_xlabel('Tokunaga parameter c')
    axes[0][1].yaxis.set_label_position("right")
    axes[0][1].yaxis.tick_right()
    axes[0][1].set_ylabel('Angle [°]')
    axes[0][1].set_xlim((0.6,5))

    axes_list = [axes[0][0],axes[0][1]]
    for ax in axes_list:  
        ax.tick_params(labelsize=size0,direction='in',width=1,length=8,pad=10)
        ax.tick_params(which='minor',direction='in')
        ax.spines['bottom'].set_linewidth(boxwidth)
        ax.spines['left'].set_linewidth(boxwidth)
        ax.spines['right'].set_linewidth(boxwidth)
        ax.spines['top'].set_linewidth(boxwidth)


    # Side-branching vs AI
    custom_bin_and_plot(dfplot1, 'meanAI_class', (0.6, 5), (32, 80), axes[1][0], palette1)
    axes[1][0].set_ylabel('Side-branching\nangle [°]', fontsize=36)

    # Side-branching vs Slope
    custom_bin_and_plot(dfplot1, 'meanSlope_class', (0.6, 5), (32, 80), axes[1][1], palette2)

    # Bifurcation vs AI
    custom_bin_and_plot(dfplot2, 'meanAI_class', (0.6, 5), (32, 80), axes[2][0], palette1)
    axes[2][0].set_xlabel('Tokunaga parameter c', fontsize=36)
    axes[2][0].set_ylabel('Bifurcation\nangle [°]', fontsize=36)

    # Bifurcation vs Slope
    custom_bin_and_plot(dfplot2, 'meanSlope_class', (0.6, 5), (32, 80), axes[2][1], palette2)
    axes[2][1].set_xlabel('Tokunaga parameter c', fontsize=36)


    # ------ set legend ----------------
    handles, labels = axes[1][0].get_legend_handles_labels()
    leg=fig.legend(
        handles, labels,
        fontsize=size0-5,
        handletextpad=0.45,
        labelspacing=0.25,
        borderaxespad=0.25,
        loc='center',  
        bbox_to_anchor=(0.32, 0.37),  
        bbox_transform=fig.transFigure,  
    )
    frame = leg.get_frame()
    frame.set_edgecolor('black')   
    frame.set_linewidth(1.5)      
    frame.set_facecolor('white')  
    frame.set_alpha(1)             


    handles, labels = axes[1][1].get_legend_handles_labels()
    leg=fig.legend(
        handles, labels,
        fontsize=size0-5,
        handletextpad=0.45,
        labelspacing=0.25,
        borderaxespad=0.25,
        loc='center',  
        bbox_to_anchor=(0.714, 0.37),  
        bbox_transform=fig.transFigure,
        frameon=True
    )

    frame = leg.get_frame()
    frame.set_edgecolor('black')   
    frame.set_linewidth(1.5)       
    frame.set_facecolor('white')   
    frame.set_alpha(1)             


    labels = ['a', 'b', 'c', 'd','e','f']
    positions = [(0, 0), (0, 1), (1, 0), (1, 1),(2,0),(2,1)]  

    for label, (i, j) in zip(labels, positions):
        axes[i][j].text(0.02, 0.97, label, transform=axes[i][j].transAxes,
                        fontsize=size0 + 8, fontweight='bold',
                        ha='left', va='top', color='black')

    axes[1][0].tick_params(axis='x', labelbottom=False)
    axes[1][1].tick_params(axis='x', labelbottom=False)
    axes[1][1].tick_params(axis='y', labelleft=False)
    axes[2][1].tick_params(axis='y', labelleft=False)

    plt.tight_layout()

    # ****************** figure 4 *******************************
    Tk_bins =[0.4,2.0,2.3,2.6,2.9,6.6]
    ecdfplot(Tk,'Tk_c', 5,  Tk_bins, 'Bifurcation')
    ecdfplot(Tk,'Tk_c', 5,  Tk_bins, 'Sidebranch')
    
    # ****************** statistics for figure 5 ****************
    df0 =Tk[['Tk_c','Sidebranch','AI','SR','Slope']]
    df0=df0.dropna()
    df0_ranked = df0.rank()
    partial_corr = pg.pcorr(df0_ranked).round(2)

    print(partial_corr)
    

    
if __name__ == '__main__':
    main()
