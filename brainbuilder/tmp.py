"""
def plot():

    out_df['y'] = out_df['slab_order']
    for resolution, tdf in out_df.groupby(['resolution']):
        for slab, sdf in tdf.groupby(['slab']):
            tdf['y'].loc[tdf['slab']==slab] = 100. * (sdf['y']-np.min(sdf['y'])) / (sdf['y'].max() - sdf['y'].min())
            print(slab)
            print(np.min(tdf['y'].loc[tdf['slab']==slab]))
            print(np.max(tdf['y'].loc[tdf['slab']==slab]))

    
    sns.set(rc={'figure.figsize':(18,18)})
    sns.set(font_scale=2)
   
    #DEBUG
    tdf['Receptor']=tdf['ligand'].apply(lambda x :ligand_receptor_dict[x] )
    sns.scatterplot(x='y', y='align_dice', hue='Receptor', data=tdf, alpha=0.3,palette='Set1')
    plt.xlabel('% Normalized distance along coronal axis within tissue slab')
    plt.ylabel('Dice Score')
    plt.title('Quantitative Validation of Histological Section Alignment')

    print(f'{qc_dir}/section_validation_{resolution}mm.png')
    plt.savefig(f'{qc_dir}/section_validation_{resolution}mm.png')
    plt.clf()
    plt.cla()
    
    print(out_df.groupby(['ligand'])['align_dice'].mean())

    if not os.path.exists(avg_csv) :
        avg_df = output_stats(out_df)
        avg_df.reset_index(inplace=True)
        avg_df.to_csv(avg_csv)
    else:
        avg_df = pd.read_csv(avg_csv)
"""
