import pathlib
path = pathlib.Path.cwd()
if path.stem == 'tmb_survival':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('tmb_survival')]
import sys
sys.path.append(str(cwd))
import pyranges as pr
import pandas as pd
import pickle

##https://www.synapse.org/#!Synapse:syn27056697
##get survival data of all patients in cohort
usecols = ['record_id', 'os_dx_status', 'tt_os_dx_days']
sample_table = pd.read_csv(open(cwd / 'files' / 'genie' / 'crc' / 'cancer_level_dataset_index.csv'), sep=',', usecols=usecols
                           )
sample_table.rename(columns={'record_id': 'bcr_patient_barcode',
                             'os_dx_status': 'OS',
                             'tt_os_dx_days': 'OS.days'}, inplace=True)

sample_table.drop_duplicates(subset=['bcr_patient_barcode'], inplace=True)


##get samples, cancer type, panels
usecols = ['record_id', 'cpt_genie_sample_id', 'cpt_oncotree_code', 'dx_path_proc_cpt_days', 'cpt_seq_assay_id', 'sample_type']
samples = pd.read_csv(open(cwd / 'files' / 'genie' / 'crc' / 'cancer_panel_test_level_dataset.csv'), sep=',', usecols=usecols)
samples.rename(columns={'record_id': 'bcr_patient_barcode',
                        'cpt_genie_sample_id': 'Tumor_Sample_Barcode',
                        'cpt_oncotree_code': 'type',
                        'cpt_seq_assay_id': 'panel'}, inplace=True)

samples.dropna(axis=0, subset='panel', inplace=True)
samples = samples.loc[samples['sample_type'] == 'Primary tumor']
samples = samples.loc[samples['dx_path_proc_cpt_days'] < 180]
samples = samples.loc[samples['type'].isin(['COAD', 'READ', 'COADREAD'])]
samples = samples.loc[~samples['panel'].isin(['VICC-01-SOLIDTUMOR', 'UHN-48-V1'])]

##only use samples that are present in the genie release 14.1
data = pd.read_csv(open(cwd / 'files' / 'genie' / 'data_clinical_sample.txt'), sep='\t', skiprows=4)
samples = samples.loc[samples['Tumor_Sample_Barcode'].isin(data['SAMPLE_ID'])]

##only use samples with treatment info
data = pd.read_csv(open(cwd / 'files' / 'genie' / 'crc' / 'regimen_cancer_level_dataset.csv'), sep=',', skiprows=0)
treatments = data[['record_id', 'regimen_drugs']].groupby('record_id').agg({'regimen_drugs': lambda x: ', '.join(x)}).reset_index()
treatments.rename(columns={'record_id': 'bcr_patient_barcode'}, inplace=True)
samples = samples.loc[samples['bcr_patient_barcode'].isin(treatments['bcr_patient_barcode'])]

##use genie release 14.1, https://www.synapse.org/#!Synapse:syn52918985
usecols = ['Variant_Classification', 'Tumor_Sample_Barcode']
data = pd.read_csv(open(cwd / 'files' / 'genie' / 'data_mutations_extended.txt'), sep='\t', skiprows=0, low_memory=False, usecols=usecols)
data = data.loc[data['Tumor_Sample_Barcode'].isin(samples['Tumor_Sample_Barcode'])]

non_syn = ['Missense_Mutation', 'Nonsense_Mutation', 'Frame_Shift_Del', 'Frame_Shift_Ins', 'In_Frame_Del', 'In_Frame_Ins', 'Nonstop_Mutation']
data = data.loc[data['Variant_Classification'].isin(non_syn)]

counts = data[['Tumor_Sample_Barcode']].groupby('Tumor_Sample_Barcode').size().to_frame('counts').reset_index()

##left merge onto samples, any nans are assumed to be 0 TMB
samples = pd.merge(samples, counts, how='left', on='Tumor_Sample_Barcode')
samples['counts'].fillna(0, inplace=True)

##calculate TMB by dividing by correct panel size

chromosomes = list(map(lambda x: str(x), list(range(1, 23)) + ['X', 'Y']))
gff = pd.read_csv(cwd / 'files' / 'Homo_sapiens.GRCh37.87.gff3',
                  sep='\t',
                  names=['chr', 'unknown', 'gene_part', 'start', 'end', 'unknown2', 'strand', 'unknown3', 'gene_info'],
                  usecols=['chr', 'gene_part', 'start', 'end', 'gene_info'],
                  low_memory=False)
gff_cds_pr = pr.PyRanges(gff.loc[(gff['gene_part'] == 'CDS') & gff['chr'].isin(chromosomes), ['chr', 'start', 'end', 'gene_info']].astype({'start': int, 'end': int}).rename(columns={'chr': 'Chromosome', 'start': 'Start', 'end': 'End'})).merge()


genie = pd.read_csv(cwd / 'files' / 'genie' / 'genomic_information.txt', sep='\t', low_memory=False)
panel_sizes = {}
for panel in samples['panel'].unique():
    panel_pr = pr.PyRanges(genie.loc[genie['SEQ_ASSAY_ID'] == panel][['Chromosome', 'Start_Position', 'End_Position']].rename(columns={'Start_Position': 'Start', 'End_Position': 'End'}))
    panel_cds_pr = panel_pr.intersect(gff_cds_pr).merge()
    panel_sizes[panel] = sum([i + 1 for i in panel_cds_pr.lengths()])

samples['tmb'] = samples['counts'] / (samples['panel'].apply(lambda x: panel_sizes[x]) / 1e6)

tmb_means = samples.groupby(['bcr_patient_barcode'])['tmb'].mean().to_frame('mean_tmb').reset_index()
tmb_panels = samples.groupby('bcr_patient_barcode')['panel'].apply(list).to_frame('panels').reset_index()
tmb_df = pd.merge(tmb_means, tmb_panels, on='bcr_patient_barcode')

##merge back to original sample table with survival
sample_table = pd.merge(sample_table, tmb_df, on='bcr_patient_barcode', how='inner')

##add regimens
sample_table = pd.merge(sample_table, treatments, on='bcr_patient_barcode', how='inner')

with open(cwd / 'figures' / 'supp_fig2' / 'crc_sample_table.pkl', 'wb') as f:
    pickle.dump(sample_table, f)


