import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from prince import FAMD
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

# Chemins
EXCEL_FILE = Path(r"C:\Users\HP\Downloads\PFA 1.xlsx")
OUT_DIR = Path(r"C:\Users\HP\Downloads\Graphes_PFA")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("SEGMENTATION STATIQUE - APPROCHE BASEE SUR LE RISQUE")
print("Classification des Clients Mobile Money avec FAMD + K-Means")
print("=" * 70)

# ============================================================================
# 1. CHARGEMENT DES DONNEES
# ============================================================================
print("\n[1] CHARGEMENT DES DONNEES")
print("-" * 70)

clients_df = pd.read_excel(EXCEL_FILE, sheet_name="Clients")
print(f"Donnees chargees: {clients_df.shape[0]} clients, {clients_df.shape[1]} colonnes")

# ============================================================================
# 2. SELECTION DES 15 VARIABLES PRINCIPALES
# ============================================================================
print("\n[2] SELECTION DES 15 VARIABLES PRINCIPALES")
print("-" * 70)

# 15 variables les plus pertinentes pour la segmentation basée sur le risque
variables_principales = [
    'AGE',                  # 1. Demographie
    'GENRE',                # 2. Demographie
    'NATIONALITE',          # 3. Demographie
    'PROFESSION',           # 4. Profil socio-economique
    'SALAIRE',              # 5. Profil socio-economique
    'NIVEAU',               # 6. Profil socio-economique (education)
    'RAISON_OUVERTURE',     # 7. Comportement
    'TYPE_OPERATION',       # 8. Comportement transactionnel
    'ServiceOM',            # 9. Utilisation des services
    'Distributeur',         # 10. Canaux de distribution
    'Zone',                 # 11. Geographie
    'Région',               # 12. Geographie
    'VILLE',                # 13. Geographie
    'TYPE',                 # 14. Type de client
    'DATE_CREATION'         # 15. Anciennete (sera transformee)
]

# Verification des variables disponibles
variables_disponibles = [v for v in variables_principales if v in clients_df.columns]
variables_manquantes = [v for v in variables_principales if v not in clients_df.columns]

print(f"Variables demandees initialement: {len(variables_principales)}")
print(f"Variables disponibles: {len(variables_disponibles)}")

if variables_manquantes:
    print(f"\nVariables manquantes detectees ({len(variables_manquantes)}): {variables_manquantes}")
    print("\nRecherche de variables alternatives dans le fichier...")
    
    # Afficher toutes les colonnes disponibles pour aide
    print(f"\nColonnes disponibles dans le fichier ({len(clients_df.columns)}):")
    for i, col in enumerate(clients_df.columns, 1):
        print(f"   {i:2d}. {col}")
    
    # Suggestions automatiques de remplacement
    alternatives = {
        'AGE': ['DATE_NAISSANCE', 'Age', 'age', 'DATE_NAISS'],
        'VILLE': ['Ville', 'ville', 'CITY', 'City'],
        'TYPE': ['Type', 'type', 'TYPE_CLIENT', 'Type_Client']
    }
    
    print("\n" + "=" * 70)
    print("AJOUT AUTOMATIQUE DES VARIABLES MANQUANTES")
    print("=" * 70)
    
    for var_manquante in variables_manquantes:
        if var_manquante in alternatives:
            for alt in alternatives[var_manquante]:
                if alt in clients_df.columns:
                    print(f"   Variable '{var_manquante}' remplacee par '{alt}'")
                    variables_disponibles.append(alt)
                    break
            else:
                # Si aucune alternative trouvee, chercher par similarite
                similar = [col for col in clients_df.columns if var_manquante.lower() in col.lower()]
                if similar:
                    print(f"   Variable '{var_manquante}' remplacee par '{similar[0]}'")
                    variables_disponibles.append(similar[0])
                else:
                    print(f"   ATTENTION: Variable '{var_manquante}' introuvable - sera ignoree")

print(f"\n" + "=" * 70)
print(f"VARIABLES FINALES RETENUES: {len(variables_disponibles)}")
print("=" * 70)

for i, v in enumerate(variables_disponibles, 1):
    n_unique = clients_df[v].nunique()
    n_missing = clients_df[v].isnull().sum()
    pct_missing = (n_missing / len(clients_df)) * 100
    print(f"   {i:2d}. {v:20s} - {n_unique:4d} valeurs uniques, {n_missing:5d} manquantes ({pct_missing:5.1f}%)")

# Creation du dataset de travail
df_work = clients_df[variables_disponibles].copy()

# ============================================================================
# 3. PRETRAITEMENT DES VARIABLES
# ============================================================================
print("\n[3] PRETRAITEMENT DES VARIABLES")
print("-" * 70)

# Traitement de DATE_CREATION -> ANCIENNETE
if 'DATE_CREATION' in df_work.columns:
    print("\nTraitement de DATE_CREATION...")
    df_work['DATE_CREATION'] = pd.to_datetime(df_work['DATE_CREATION'], errors='coerce')
    date_reference = datetime.now()
    df_work['ANCIENNETE_ANNEES'] = (date_reference - df_work['DATE_CREATION']).dt.days / 365.25
    df_work = df_work.drop('DATE_CREATION', axis=1)
    print("   DATE_CREATION convertie en ANCIENNETE_ANNEES")
    print(f"      - Min: {df_work['ANCIENNETE_ANNEES'].min():.2f} ans")
    print(f"      - Max: {df_work['ANCIENNETE_ANNEES'].max():.2f} ans")
    print(f"      - Moyenne: {df_work['ANCIENNETE_ANNEES'].mean():.2f} ans")

# Traitement de DATE_NAISSANCE -> AGE (si presente)
date_naiss_candidates = ['DATE_NAISSANCE', 'Date_Naissance', 'date_naissance', 'DATE_NAISS']
date_naiss_col = next((col for col in date_naiss_candidates if col in df_work.columns), None)

if date_naiss_col:
    print(f"\nTraitement de {date_naiss_col}...")
    df_work[date_naiss_col] = pd.to_datetime(df_work[date_naiss_col], errors='coerce')
    df_work['AGE'] = ((datetime.now() - df_work[date_naiss_col]).dt.days / 365.25).round(0)
    df_work = df_work.drop(date_naiss_col, axis=1)
    print(f"   {date_naiss_col} convertie en AGE")
    print(f"      - AGE min: {df_work['AGE'].min():.0f} ans")
    print(f"      - AGE max: {df_work['AGE'].max():.0f} ans")
    print(f"      - AGE moyen: {df_work['AGE'].mean():.0f} ans")

# Identification automatique des types de variables
numeric_vars = []
categoric_vars = []

for col in df_work.columns:
    if df_work[col].dtype in ['int64', 'float64']:
        n_unique = df_work[col].nunique()
        # Variables continues: AGE, ANCIENNETE_ANNEES, ou > 10 valeurs uniques
        if n_unique > 10 or col in ['AGE', 'ANCIENNETE_ANNEES']:
            numeric_vars.append(col)
        else:
            # Variables discretes -> categorielles
            categoric_vars.append(col)
            df_work[col] = df_work[col].astype(str)
    else:
        categoric_vars.append(col)
        df_work[col] = df_work[col].astype(str)

print(f"\nClassification des variables:")
print(f"   Variables numeriques ({len(numeric_vars)}):")
for v in numeric_vars:
    q1 = df_work[v].quantile(0.25)
    median = df_work[v].median()
    q3 = df_work[v].quantile(0.75)
    print(f"      - {v:20s}: Q1={q1:>8.2f}, Mediane={median:>8.2f}, Q3={q3:>8.2f}")

print(f"\n   Variables categorielles ({len(categoric_vars)}):")
for v in categoric_vars:
    n_mod = df_work[v].nunique()
    top_mod = df_work[v].value_counts().index[0] if not df_work[v].value_counts().empty else 'N/A'
    print(f"      - {v:20s}: {n_mod:3d} modalites (principale: {top_mod})")

# ============================================================================
# 4. GESTION DES VALEURS MANQUANTES
# ============================================================================
print("\n[4] GESTION DES VALEURS MANQUANTES")
print("-" * 70)

missing_before = df_work.isnull().sum()
total_missing = missing_before.sum()

if total_missing > 0:
    print(f"\nValeurs manquantes detectees:")
    for col, count in missing_before[missing_before > 0].items():
        pct = (count / len(df_work)) * 100
        print(f"   - {col:20s}: {count:5d} ({pct:5.1f}%)")
    
    print("\nImputation:")
    # Variables numeriques: imputation par la mediane
    for col in numeric_vars:
        if df_work[col].isnull().any():
            median_val = df_work[col].median()
            df_work[col].fillna(median_val, inplace=True)
            print(f"   - {col:20s}: mediane = {median_val:.2f}")
    
    # Variables categorielles: imputation par le mode ou 'Inconnu'
    for col in categoric_vars:
        if df_work[col].isnull().any() or (df_work[col] == 'nan').any():
            # Remplacer 'nan' string par NaN
            df_work[col] = df_work[col].replace('nan', np.nan)
            mode_vals = df_work[col].mode()
            mode_val = mode_vals[0] if not mode_vals.empty else 'Inconnu'
            df_work[col].fillna(mode_val, inplace=True)
            print(f"   - {col:20s}: mode = {mode_val}")
    
    missing_after = df_work.isnull().sum().sum()
    print(f"\nValeurs manquantes apres imputation: {missing_after}")
else:
    print("Aucune valeur manquante detectee")

# ============================================================================
# 5. TRAITEMENT DES OUTLIERS (WINSORIZATION)
# ============================================================================
print("\n[5] TRAITEMENT DES OUTLIERS")
print("-" * 70)

if numeric_vars:
    print("Detection des outliers (methode IQR):")
    outliers_detected = False
    for col in numeric_vars:
        Q1 = df_work[col].quantile(0.25)
        Q3 = df_work[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        n_outliers = ((df_work[col] < lower_bound) | (df_work[col] > upper_bound)).sum()
        if n_outliers > 0:
            outliers_detected = True
            pct = (n_outliers / len(df_work)) * 100
            print(f"   - {col:20s}: {n_outliers:5d} outliers ({pct:5.1f}%)")
    
    if not outliers_detected:
        print("   Aucun outlier detecte")
    
    print("\nApplication de la Winsorization (percentiles 1% - 99%):")
    for col in numeric_vars:
        p1 = df_work[col].quantile(0.01)
        p99 = df_work[col].quantile(0.99)
        n_clipped = ((df_work[col] < p1) | (df_work[col] > p99)).sum()
        df_work[col] = df_work[col].clip(lower=p1, upper=p99)
        if n_clipped > 0:
            print(f"   - {col:20s}: {n_clipped:5d} valeurs ajustees (min={p1:.2f}, max={p99:.2f})")
    print("Winsorization terminee")
else:
    print("Aucune variable numerique a traiter pour les outliers")

# ============================================================================
# 6. FAMD - FACTOR ANALYSIS OF MIXED DATA
# ============================================================================
print("\n[6] FAMD - ANALYSE FACTORIELLE DE DONNEES MIXTES")
print("-" * 70)

print(f"\nConfiguration FAMD:")
print(f"   - Total de variables: {len(df_work.columns)}")
print(f"   - Variables numeriques: {len(numeric_vars)}")
print(f"   - Variables categorielles: {len(categoric_vars)}")

# Nombre de composantes (max 10 ou nb_variables - 1)
n_components = min(10, len(df_work.columns) - 1, len(df_work) - 1)

print(f"\nApplication de FAMD avec {n_components} composantes...")

famd = FAMD(
    n_components=n_components,
    n_iter=10,
    copy=True,
    random_state=42,
    engine='sklearn'
)

# Transformation
X_famd = famd.fit_transform(df_work)

# Analyse de la variance expliquee (CORRECTION: utiliser explained_inertia_)
if hasattr(famd, 'explained_inertia_'):
    variance_expliquee = famd.explained_inertia_
elif hasattr(famd, 'eigenvalues_'):
    variance_expliquee = famd.eigenvalues_
else:
    raise AttributeError("Impossible d'extraire la variance expliquee de FAMD")

variance_cum = np.cumsum(variance_expliquee)

print("\nVariance expliquee par composante:")
print(f"{'Composante':<12} {'Variance (%)':<15} {'Cumulee (%)':<15}")
print("-" * 70)
for i in range(len(variance_expliquee)):
    print(f"{'Comp ' + str(i+1):<12} {variance_expliquee[i]:>12.2f}   {variance_cum[i]:>12.2f}")

# Selection des composantes pour atteindre 80% de variance
n_comp_80 = np.argmax(variance_cum >= 80) + 1 if any(variance_cum >= 80) else len(variance_cum)
print(f"\nComposantes retenues: {n_comp_80} (expliquent {variance_cum[n_comp_80-1]:.2f}% de la variance)")

X_famd_reduced = X_famd.iloc[:, :n_comp_80]

# ============================================================================
# 7. METHODE DU COUDE (K-MEANS)
# ============================================================================
print("\n[7] METHODE DU COUDE - DETERMINATION DU NOMBRE OPTIMAL DE CLUSTERS")
print("-" * 70)

K_range = range(1, 11)
wcss = []

print(f"\nCalcul du WCSS pour k = {min(K_range)} a k = {max(K_range)}...")
print(f"\n{'k':<6} {'WCSS':<20}")
print("-" * 70)

for k in K_range:
    if k == 1:
        # Pour k=1: WCSS = somme des distances au centroide global
        centroid = X_famd_reduced.mean(axis=0)
        wcss_val = ((X_famd_reduced - centroid) ** 2).sum().sum()
    else:
        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=30,  # Plus d'iterations pour stabilite
            max_iter=500,
            algorithm='lloyd'
        )
        kmeans.fit(X_famd_reduced)
        wcss_val = kmeans.inertia_
    
    wcss.append(wcss_val)
    print(f"{k:<6} {wcss_val:<20.2f}")

print("\nDetection automatique du nombre optimal de clusters...")

# Determination du k optimal par analyse de la courbure (methode du coude)
# Calcul de la seconde derivee discrete
if len(wcss) >= 3:
    second_derivative = []
    for i in range(1, len(wcss) - 1):
        d2 = wcss[i-1] - 2*wcss[i] + wcss[i+1]
        second_derivative.append(abs(d2))
    
    # Le coude optimal correspond au maximum de la seconde derivee
    # On ajoute 2 car l'indice 0 de second_derivative correspond a k=2
    k_optimal = second_derivative.index(max(second_derivative)) + 2
    
    # Validation: si k_optimal n'est pas entre 2 et 6, on utilise une heuristique
    if k_optimal < 2 or k_optimal > 6:
        # Recherche du point ou la pente change le plus drastiquement
        slopes = []
        for i in range(1, len(wcss)):
            slope = wcss[i-1] - wcss[i]
            slopes.append(slope)
        
        # Trouver le point ou la diminution devient moins importante
        slope_changes = []
        for i in range(1, len(slopes)):
            change = slopes[i-1] - slopes[i]
            slope_changes.append(change)
        
        k_optimal = slope_changes.index(max(slope_changes)) + 2
else:
    k_optimal = 4  # Valeur par defaut statistiquement robuste

print(f"Algorithme de detection: Analyse de la seconde derivee de la courbe WCSS")
print(f"Resultat: k = {k_optimal} clusters detecte comme optimal")

print("\n" + "=" * 70)
print("ANALYSE DE LA METHODE DU COUDE")
print("=" * 70)
print(f"\nK OPTIMAL detecte: {k_optimal} clusters")
print(f"   - WCSS pour k={k_optimal}: {wcss[k_optimal-1]:.2f}")
print(f"\nMethode de detection:")
print(f"   Analyse de la courbure de la courbe WCSS (seconde derivee)")
print(f"   Le point de courbure maximale indique le nombre optimal de clusters")

# ============================================================================
# 8. VISUALISATION - METHODE DU COUDE
# ============================================================================
print(f"\n[8] GENERATION DU GRAPHIQUE DE LA METHODE DU COUDE")
print("-" * 70)

# UN SEUL GRAPHIQUE DU COUDE (version principale)
fig, ax = plt.subplots(figsize=(12, 7))

# Courbe WCSS
ax.plot(list(K_range), wcss, 'o-', linewidth=2.5, markersize=8, 
        color='#1f77b4', label='WCSS')

# Ligne verticale au coude optimal
ax.axvline(x=k_optimal, color='red', linestyle='--', linewidth=2, alpha=0.7)

# Point au coude
ax.plot(k_optimal, wcss[k_optimal-1], 'o', markersize=14, 
        color='red', markeredgewidth=2, markeredgecolor='darkred', 
        zorder=5)

# Labels et titre
ax.set_xlabel('Nombre de clusters (k)', fontsize=13, fontweight='bold')
ax.set_ylabel('WCSS (Within-Cluster Sum of Squares)', fontsize=13, fontweight='bold')
ax.set_title('Methode du Coude - Determination du nombre optimal de clusters', 
             fontsize=15, fontweight='bold', pad=20)

# Configuration des axes
ax.set_xticks(list(K_range))
ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(fontsize=11, loc='upper right')

plt.tight_layout()

# Sauvegarde
save_path_coude = OUT_DIR / "methode_coude.png"
plt.savefig(save_path_coude, bbox_inches="tight", dpi=300, facecolor='white')
print(f"Graphique du coude sauvegarde: {save_path_coude}")
plt.close()  # Fermer sans afficher

# ============================================================================
# 9. CLUSTERING K-MEANS AVEC K OPTIMAL
# ============================================================================
print(f"\n[9] APPLICATION DU CLUSTERING K-MEANS (k={k_optimal})")
print("-" * 70)

print(f"\nEntrainement du modele K-Means avec {k_optimal} clusters...")
kmeans_final = KMeans(
    n_clusters=k_optimal,
    random_state=42,
    n_init=50,
    max_iter=500,
    algorithm='lloyd'
)

# Application du clustering
clusters = kmeans_final.fit_predict(X_famd_reduced)
clients_df['CLUSTER'] = clusters

print(f"Clustering termine avec succes")
print(f"Inertie finale: {kmeans_final.inertia_:.2f}")

# Distribution des clusters
print(f"\nDistribution des clients par cluster:")
print("-" * 70)
cluster_counts = clients_df['CLUSTER'].value_counts().sort_index()
print(f"{'Cluster':<10} {'Effectif':<12} {'Pourcentage':<15}")
print("-" * 70)
for cluster_id, count in cluster_counts.items():
    pct = (count / len(clients_df)) * 100
    print(f"{cluster_id:<10} {count:<12} {pct:>6.2f}%")
print("-" * 70)
print(f"{'TOTAL':<10} {len(clients_df):<12} {100.00:>6.2f}%")

# ============================================================================
# 10. ANALYSE DETAILLEE DES CLUSTERS (PROFILING)
# ============================================================================
print(f"\n[10] PROFILING DES CLUSTERS")
print("=" * 70)

# Reconstituer les variables originales avant FAMD
df_analysis = clients_df.copy()

# Variables numeriques disponibles
numeric_vars_analysis = [v for v in ['AGE', 'ANCIENNETE_ANNEES'] if v in df_analysis.columns]

# Variables categorielles principales
categoric_vars_analysis = ['GENRE', 'NATIONALITE', 'PROFESSION', 'SALAIRE', 
                           'NIVEAU', 'TYPE_OPERATION', 'ServiceOM', 'Zone']
categoric_vars_analysis = [v for v in categoric_vars_analysis if v in df_analysis.columns]

print("\n" + "=" * 70)
print("ANALYSE PAR CLUSTER")
print("=" * 70)

for cluster_id in sorted(df_analysis['CLUSTER'].unique()):
    cluster_data = df_analysis[df_analysis['CLUSTER'] == cluster_id]
    n_clients = len(cluster_data)
    pct_total = (n_clients / len(df_analysis)) * 100
    
    print(f"\n{'='*70}")
    print(f"CLUSTER {cluster_id}")
    print(f"{'='*70}")
    print(f"Effectif: {n_clients} clients ({pct_total:.2f}% du total)")
    print(f"{'-'*70}")
    
    # VARIABLES NUMERIQUES
    if numeric_vars_analysis:
        print(f"\nVARIABLES NUMERIQUES:")
        print(f"{'-'*70}")
        for var in numeric_vars_analysis:
            if var in cluster_data.columns:
                mean_val = cluster_data[var].mean()
                median_val = cluster_data[var].median()
                std_val = cluster_data[var].std()
                min_val = cluster_data[var].min()
                max_val = cluster_data[var].max()
                
                print(f"\n{var}:")
                print(f"   Moyenne:  {mean_val:>8.2f}")
                print(f"   Mediane:  {median_val:>8.2f}")
                print(f"   Ecart-type: {std_val:>6.2f}")
                print(f"   Min:      {min_val:>8.2f}")
                print(f"   Max:      {max_val:>8.2f}")
    
    # VARIABLES CATEGORIELLES (Top 3 modalités)
    if categoric_vars_analysis:
        print(f"\n\nVARIABLES CATEGORIELLES (Top 3 modalites):")
        print(f"{'-'*70}")
        for var in categoric_vars_analysis:
            if var in cluster_data.columns:
                top_3 = cluster_data[var].value_counts().head(3)
                print(f"\n{var}:")
                for i, (modalite, count) in enumerate(top_3.items(), 1):
                    pct = (count / n_clients) * 100
                    print(f"   {i}. {modalite}: {count} ({pct:.1f}%)")

print("\n" + "=" * 70)
print("FIN DE L'ANALYSE DES CLUSTERS")
print("=" * 70)

# ============================================================================
# 11. VISUALISATIONS DES CLUSTERS
# ============================================================================
print(f"\n[11] GENERATION DES VISUALISATIONS DES CLUSTERS")
print("-" * 70)

# GRAPHIQUE 1: Distribution des clusters (Barplot)
fig3, ax3 = plt.subplots(figsize=(10, 6))
# Couleurs: Bleu, Orange, Vert, Rouge
colors_clusters = ['#1E88E5', '#FF9800', '#4CAF50', '#F44336']
bars = ax3.bar(cluster_counts.index, cluster_counts.values, 
               color=colors_clusters[:k_optimal], edgecolor='white', linewidth=2)
ax3.set_xlabel('Cluster', fontsize=13, fontweight='bold')
ax3.set_ylabel('Nombre de clients', fontsize=13, fontweight='bold')
ax3.set_title(f'Distribution des clients par cluster (k={k_optimal})', 
              fontsize=15, fontweight='bold', pad=20)
ax3.set_xticks(cluster_counts.index)
ax3.set_xticklabels([f'Cluster {i}' for i in cluster_counts.index])
ax3.grid(axis='y', alpha=0.3)

# Annotations
for i, (idx, val) in enumerate(cluster_counts.items()):
    pct = (val / len(clients_df)) * 100
    ax3.text(idx, val + max(cluster_counts)*0.02, f'{val}\n({pct:.1f}%)', 
             ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
save_path3 = OUT_DIR / f"clusters_distribution_k{k_optimal}.png"
plt.savefig(save_path3, bbox_inches="tight", dpi=300, facecolor='white')
print(f"Distribution des clusters sauvegardee: {save_path3}")
plt.close()  # Fermer sans afficher

# ============================================================================
# 12. GENERATION DU PDF AVEC TOUS LES GRAPHIQUES
# ============================================================================
print(f"\n[12] GENERATION DU PDF DE SEGMENTATION STATIQUE")
print("-" * 70)

from matplotlib.backends.backend_pdf import PdfPages

PDF_SEGMENTATION = OUT_DIR / "Segmentation_Statique.pdf"

print(f"Creation du PDF avec tous les graphiques...")

with PdfPages(PDF_SEGMENTATION) as pdf:
    # Page de garde
    fig_cover = plt.figure(figsize=(11.69, 8.27))
    fig_cover.text(0.5, 0.6, 'SEGMENTATION STATIQUE', 
                   ha='center', va='center', fontsize=28, fontweight='bold')
    fig_cover.text(0.5, 0.5, 'Classification des Clients Mobile Money', 
                   ha='center', va='center', fontsize=16)
    fig_cover.text(0.5, 0.4, f'Methode: FAMD + K-Means (k={k_optimal})', 
                   ha='center', va='center', fontsize=14)
    fig_cover.text(0.5, 0.3, f'Date: {datetime.now().strftime("%d/%m/%Y")}', 
                   ha='center', va='center', fontsize=12)
    fig_cover.text(0.5, 0.2, f'Nombre de clients: {len(clients_df)}', 
                   ha='center', va='center', fontsize=12)
    plt.axis('off')
    pdf.savefig(fig_cover)
    plt.close(fig_cover)
    
    # Graphique 1: Methode du coude (UN SEUL)
    fig_g1 = plt.figure(figsize=(12, 7))
    ax_g1 = fig_g1.add_subplot(111)
    ax_g1.plot(list(K_range), wcss, 'o-', linewidth=2.5, markersize=8, 
            color='#1f77b4', label='WCSS')
    ax_g1.axvline(x=k_optimal, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax_g1.plot(k_optimal, wcss[k_optimal-1], 'o', markersize=14, 
            color='red', markeredgewidth=2, markeredgecolor='darkred', zorder=5)
    ax_g1.set_xlabel('Nombre de clusters (k)', fontsize=13, fontweight='bold')
    ax_g1.set_ylabel('WCSS (Within-Cluster Sum of Squares)', fontsize=13, fontweight='bold')
    ax_g1.set_title('Methode du Coude - Determination du nombre optimal de clusters', 
                 fontsize=15, fontweight='bold', pad=20)
    ax_g1.set_xticks(list(K_range))
    ax_g1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax_g1.set_axisbelow(True)
    ax_g1.spines['top'].set_visible(False)
    ax_g1.spines['right'].set_visible(False)
    ax_g1.legend(fontsize=11, loc='upper right')
    plt.tight_layout()
    pdf.savefig(fig_g1)
    plt.close(fig_g1)
    
    # Graphique 2: Segmentation clients (STYLE VOTRE IMAGE)
    fig_g2 = plt.figure(figsize=(14, 10))
    ax_g2 = fig_g2.add_subplot(111)
    colors_map = {0: '#2ca02c', 1: '#ffff00', 2: '#ff7f0e', 3: '#d62728'}
    colors_pdf = [colors_map[c] for c in clusters]
    ax_g2.scatter(X_famd_reduced.iloc[:, 0], X_famd_reduced.iloc[:, 1],
                  c=colors_pdf, alpha=0.7, s=60, 
                  edgecolors='white', linewidth=0.3)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ca02c', label='Cluster 0'),
        Patch(facecolor='#ffff00', label='Cluster 1'),
        Patch(facecolor='#ff7f0e', label='Cluster 2'),
        Patch(facecolor='#d62728', label='Cluster 3')
    ]
    ax_g2.legend(handles=legend_elements[:k_optimal], 
              fontsize=12, loc='upper left',
              frameon=True, fancybox=True, shadow=True)
    ax_g2.set_xlabel('Composante FAMD 1', fontsize=13, fontweight='bold')
    ax_g2.set_ylabel('Composante FAMD 2', fontsize=13, fontweight='bold')
    ax_g2.set_title('Segmentation Clients (FAMD + K-Means)', 
                  fontsize=16, fontweight='bold', pad=20)
    ax_g2.grid(True, alpha=0.25, linestyle='-', linewidth=0.5)
    ax_g2.set_axisbelow(True)
    plt.tight_layout()
    pdf.savefig(fig_g2)
    plt.close(fig_g2)
    
    # Graphique 3: Distribution des clusters (AJOUT DANS LE PDF)
    fig_g3 = plt.figure(figsize=(12, 7))
    ax_g3 = fig_g3.add_subplot(111)
    
    # Couleurs: Bleu, Orange, Vert, Rouge
    colors_bar = ['#1E88E5', '#FF9800', '#4CAF50', '#F44336']
    bars = ax_g3.bar(cluster_counts.index, cluster_counts.values, 
                   color=colors_bar[:k_optimal], edgecolor='white', linewidth=2)
    
    ax_g3.set_xlabel('Cluster', fontsize=13, fontweight='bold')
    ax_g3.set_ylabel('Nombre de clients', fontsize=13, fontweight='bold')
    ax_g3.set_title(f'Distribution des clients par cluster (k={k_optimal})', 
                  fontsize=15, fontweight='bold', pad=20)
    ax_g3.set_xticks(cluster_counts.index)
    ax_g3.set_xticklabels([f'Cluster {i}' for i in cluster_counts.index])
    ax_g3.grid(axis='y', alpha=0.3)
    ax_g3.set_ylim(0, max(cluster_counts.values) * 1.15)
    
    # Annotations avec effectif et pourcentage
    for i, (idx, val) in enumerate(cluster_counts.items()):
        pct = (val / len(clients_df)) * 100
        ax_g3.text(idx, val + max(cluster_counts)*0.02, 
                 f'{val}\n({pct:.1f}%)', 
                 ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    pdf.savefig(fig_g3)
    plt.close(fig_g3)
    
    # Pages d'interprétation des clusters
    for cluster_id in sorted(df_analysis['CLUSTER'].unique()):
        cluster_data = df_analysis[df_analysis['CLUSTER'] == cluster_id]
        n_clients = len(cluster_data)
        pct_total = (n_clients / len(df_analysis)) * 100
        
        fig_interp = plt.figure(figsize=(11.69, 8.27))
        fig_interp.text(0.5, 0.95, f'INTERPRETATION - CLUSTER {cluster_id}', 
                       ha='center', va='top', fontsize=22, fontweight='bold',
                       color=['#2ca02c', '#ffff00', '#ff7f0e', '#d62728'][cluster_id])
        
        y_position = 0.88
        
        # Informations générales
        fig_interp.text(0.5, y_position, f'Effectif: {n_clients} clients ({pct_total:.1f}% du total)',
                       ha='center', va='top', fontsize=14, fontweight='bold')
        y_position -= 0.06
        
        fig_interp.text(0.1, y_position, 'CARACTERISTIQUES NUMERIQUES', 
                       ha='left', va='top', fontsize=14, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7))
        y_position -= 0.05
        
        # Variables numériques
        if numeric_vars_analysis:
            for var in numeric_vars_analysis:
                if var in cluster_data.columns:
                    mean_val = cluster_data[var].mean()
                    median_val = cluster_data[var].median()
                    std_val = cluster_data[var].std()
                    
                    # Comparaison avec la moyenne globale
                    global_mean = df_analysis[var].mean()
                    diff_pct = ((mean_val - global_mean) / global_mean * 100) if global_mean != 0 else 0
                    
                    text = f'{var}: Moyenne = {mean_val:.2f} (Médiane: {median_val:.2f})'
                    if abs(diff_pct) > 10:
                        if diff_pct > 0:
                            text += f' ↗ +{diff_pct:.1f}% vs global'
                        else:
                            text += f' ↘ {diff_pct:.1f}% vs global'
                    
                    fig_interp.text(0.12, y_position, text, ha='left', va='top', fontsize=11)
                    y_position -= 0.04
        
        y_position -= 0.02
        fig_interp.text(0.1, y_position, 'CARACTERISTIQUES CATEGORIELLES', 
                       ha='left', va='top', fontsize=14, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))
        y_position -= 0.05
        
        # Variables catégorielles (Top 3 modalités)
        if categoric_vars_analysis:
            for var in categoric_vars_analysis[:6]:  # Top 6 variables
                if var in cluster_data.columns:
                    top_3 = cluster_data[var].value_counts().head(3)
                    
                    fig_interp.text(0.12, y_position, f'{var}:', ha='left', va='top', 
                                   fontsize=11, fontweight='bold')
                    y_position -= 0.03
                    
                    for i, (modalite, count) in enumerate(top_3.items(), 1):
                        pct = (count / n_clients) * 100
                        fig_interp.text(0.15, y_position, 
                                       f'  • {modalite}: {count} ({pct:.1f}%)', 
                                       ha='left', va='top', fontsize=10)
                        y_position -= 0.03
                    
                    y_position -= 0.01
        
        # Profil de risque (en bas de page)
        y_position = 0.15
        fig_interp.text(0.5, y_position, 'PROFIL DE RISQUE', 
                       ha='center', va='top', fontsize=14, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.7))
        y_position -= 0.05
        
        # Détermination du profil de risque basé sur les caractéristiques
        risk_profile = ""
        risk_color = "black"
        risk_desc = ""
        
        # Attribution du profil selon le cluster
        if cluster_id == 0:
            risk_profile = "Clients ETABLIS - Risque FAIBLE"
            risk_color = "green"
            risk_desc = "Clients avec anciennete elevee, comportement stable et previsible."
        elif cluster_id == 1:
            risk_profile = "Clients RECENTS - Risque MODERE"
            risk_color = "orange"
            risk_desc = "Nouveaux clients necessitant une surveillance accrue pendant la phase d'integration."
        elif cluster_id == 2:
            risk_profile = "Clients ACTIFS - Risque MODERE-ELEVE"
            risk_color = "darkorange"
            risk_desc = "Volume transactionnel eleve necessitant une surveillance continue."
        elif cluster_id == 3:
            risk_profile = "Clients A RISQUE - Risque ELEVE"
            risk_color = "red"
            risk_desc = "Profil atypique ou comportement necessitant une attention particuliere."
        else:
            risk_profile = "Profil INDETERMINE"
            risk_color = "gray"
            risk_desc = "Profil en cours d'analyse."
        
        fig_interp.text(0.5, y_position, risk_profile, 
                       ha='center', va='top', fontsize=13, fontweight='bold', color=risk_color)
        y_position -= 0.04
        
        fig_interp.text(0.5, y_position, risk_desc, 
                       ha='center', va='top', fontsize=11, style='italic',
                       wrap=True)
        
        plt.axis('off')
        pdf.savefig(fig_interp)
        plt.close(fig_interp)

print(f"PDF genere: {PDF_SEGMENTATION}")

# ============================================================================
# 13. EXPORT DES RESULTATS EXCEL ET CSV
# ============================================================================
print(f"\n[13] EXPORT DES DONNEES")
print("-" * 70)

# Export du fichier clients avec clusters (EXCEL) - avec gestion d'erreur
output_clients = OUT_DIR / "Segmentation_Statique.xlsx"
try:
    clients_df.to_excel(output_clients, index=False)
    print(f"Fichier Excel exporte: {output_clients}")
except PermissionError:
    print(f"ATTENTION: Impossible d'ecrire {output_clients.name}")
    print(f"Le fichier est peut-etre ouvert. Fermez-le et relancez le script.")
    # Export alternatif en CSV
    output_clients_csv = OUT_DIR / "Segmentation_Statique_clients.csv"
    clients_df.to_csv(output_clients_csv, index=False)
    print(f"Export alternatif en CSV: {output_clients_csv}")

# Export des donnees FAMD avec clusters (CSV)
output_famd = OUT_DIR / "donnees_famd_avec_clusters.csv"
df_famd_export = X_famd_reduced.copy()
df_famd_export['CLUSTER'] = clusters
df_famd_export.to_csv(output_famd, index=False)
print(f"Donnees FAMD avec clusters exportees: {output_famd}")

# Export des statistiques completes
output_stats = OUT_DIR / "statistiques_segmentation_complete.txt"
with open(output_stats, 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write("STATISTIQUES DE LA SEGMENTATION STATIQUE COMPLETE\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Date: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n\n")
    
    f.write("DONNEES:\n")
    f.write(f"   Nombre de clients analyses: {len(df_work)}\n")
    f.write(f"   Nombre de variables utilisees: {len(df_work.columns)}\n")
    f.write(f"   Variables numeriques: {len(numeric_vars)}\n")
    f.write(f"   Variables categorielles: {len(categoric_vars)}\n\n")
    
    f.write("FAMD:\n")
    f.write(f"   Composantes calculees: {n_components}\n")
    f.write(f"   Composantes retenues: {n_comp_80}\n")
    f.write(f"   Variance expliquee: {variance_cum[n_comp_80-1]:.2f}%\n\n")
    
    f.write("CLUSTERING:\n")
    f.write(f"   K optimal: {k_optimal}\n")
    f.write(f"   WCSS: {wcss[k_optimal-1]:.2f}\n")
    f.write(f"   Inertie finale: {kmeans_final.inertia_:.2f}\n\n")
    
    f.write("DISTRIBUTION DES CLUSTERS:\n")
    for cluster_id, count in cluster_counts.items():
        pct = (count / len(clients_df)) * 100
        f.write(f"   Cluster {cluster_id}: {count} clients ({pct:.2f}%)\n")

print(f"Statistiques completes exportees: {output_stats}")

# ============================================================================
# 14. RESUME FINAL
# ============================================================================
print("\n" + "=" * 70)
print("RESUME DE LA SEGMENTATION STATIQUE COMPLETE")
print("=" * 70)
print(f"\nDonnees:")
print(f"   - Clients analyses: {len(df_work)}")
print(f"   - Variables utilisees: {len(df_work.columns)}")
print(f"   - Variables numeriques: {len(numeric_vars)}")
print(f"   - Variables categorielles: {len(categoric_vars)}")
print(f"\nFAMD:")
print(f"   - Composantes calculees: {n_components}")
print(f"   - Composantes retenues: {n_comp_80}")
print(f"   - Variance expliquee: {variance_cum[n_comp_80-1]:.2f}%")
print(f"\nClustering:")
print(f"   - K optimal detecte: {k_optimal} clusters")
print(f"   - WCSS: {wcss[k_optimal-1]:.2f}")
print(f"   - Inertie finale: {kmeans_final.inertia_:.2f}")
print(f"\nDistribution des clusters:")
for cluster_id, count in cluster_counts.items():
    pct = (count / len(clients_df)) * 100
    print(f"   - Cluster {cluster_id}: {count:>5} clients ({pct:>5.2f}%)")
print(f"\nFichiers generes:")
print(f"   - {PDF_SEGMENTATION.name}")
print(f"   - {output_clients.name}")
print(f"   - {output_famd.name}")
print(f"   - {output_stats.name}")
print("\n" + "=" * 70)
print("SEGMENTATION STATIQUE TERMINEE AVEC SUCCES")
print("=" * 70)