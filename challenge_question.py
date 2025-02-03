from urielplus import urielplus
from sklearn.decomposition import PCA

av_agg = urielplus.URIELPlus()
un_agg = urielplus.URIELPlus()

av_agg.reset()
un_agg.reset()

#Configuration
av_agg.set_cache(True)
un_agg.set_cache(True)

#Integrating databases
av_agg.integrate_databases()
un_agg.integrate_databases()

#Imputation
av_agg.softimpute_imputation()
un_agg.softimpute_imputation()

#Aggregation
av_agg.aggregation = 'A'
av_agg.aggregate()
un_agg.aggregation = 'U'
un_agg.aggregate()

# Principal Component Analysis

def num_features_for_variance(u: urielplus.URIELPlus, variance_threshold: float = 0.95):
    pca = PCA()
    pca.fit(u.data.toarray() if hasattr(u.data, 'toarray') else u.data)
    cumulative_variance = pca.explained_variance_ratio_.cumsum()
    num_features = (cumulative_variance < variance_threshold).sum() + 1
    return num_features

num_features_agg = num_features_for_variance(av_agg)
num_features_un_agg = num_features_for_variance(un_agg)

print(f"Number of features to explain 95% variance for aggregated data: {num_features_agg}")
print(f"Number of features to explain 95% variance for unaggregated data: {num_features_un_agg}")

