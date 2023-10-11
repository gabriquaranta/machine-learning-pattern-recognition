from models import GMM
from library import utils
from library.plot_minDCF import plot_minDCF_gmm


# # GMM
def train_gmm(DTR, LTR):
    priors = [0.5, 0.1, 0.9]
    model = GMM.GMM()
    DTRpca = DTR

    print("Plotting minDCF graphs ...")
    components = [2, 4, 8, 16, 32]
    for type in ["full", "tied", "diag"]:
        for i in range(2):  # raw, pca11
            y5, y1, y9 = [], [], []
            title = "raw"
            if i > 0:
                PCA_ = utils.PCA(DTR, DTR.shape[0] - i)
                DTRpca = PCA_[0]
                title = f"pca{DTR.shape[0] - i}"
            for c in components:
                y5.append(utils.kfolds(DTRpca, LTR, priors[0], model, (c, type))[0])
                y1.append(utils.kfolds(DTRpca, LTR, priors[1], model, (c, type))[0])
                y9.append(utils.kfolds(DTRpca, LTR, priors[2], model, (c, type))[0])
            plot_minDCF_gmm(
                components, y5, y1, y9, f"{type}_{title}", f"gmm {type}-cov / {title}"
            )
    print("Done.")

    # minDCF considerations for components number:
    # gmm full -> both raw and pca min is around 4 component
    # gmm diag -> in both raw and pca min for 0.1 is at for
    #             but for target 0.5 and 0.9 is at 16
    # gmm tied -> both raw and pca min is around 4

    print("# # 5-folds")
    for i in range(2):  # raw, pca11
        print(f"# PCA m = {DTR.shape[0] - i}" if i > 0 else "# RAW")
        if i > 0:
            PCA_ = utils.PCA(DTR, DTR.shape[0] - i)
            DTRpca = PCA_[0]
        for pi in priors:
            print(f" # Prior = {pi}")

            gmm_full_c = 4
            minDCF = utils.kfolds(DTRpca, LTR, pi, model, (gmm_full_c, "full"))[0]
            print(f"  GMM Full ({gmm_full_c} components)  -> minDCF = %.3f" % minDCF)

            gmm_diag_c = 16
            minDCF = utils.kfolds(DTRpca, LTR, pi, model, (gmm_diag_c, "diag"))[0]
            print(f"  GMM Diag ({gmm_diag_c} components) -> minDCF = %.3f" % minDCF)

            gmm_tied_c = 4
            minDCF = utils.kfolds(DTRpca, LTR, pi, model, (gmm_tied_c, "tied"))[0]
            print(f"  GMM Tied ({gmm_tied_c} components) -> minDCF = %.3f" % minDCF)
