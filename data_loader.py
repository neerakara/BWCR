import data_placenta
import data_prostate

def load_data(args, sub_dataset, train_test_val):

    if args.dataset == 'placenta':
        data_path = '/data/vision/polina/projects/fetal/projects/placenta-segmentation/data/split-nifti-processed/'
        data = data_placenta.load_images_and_labels(data_path,
                                                    train_test_val = train_test_val,
                                                    cv_fold = args.cv_fold_num)

    elif args.dataset == 'prostate':
        data_path = '/data/vision/polina/users/nkarani/data/segmentation/prostate/'
        data = data_prostate.load_images_and_labels(data_path,
                                                    sub_dataset,
                                                    train_test_val = train_test_val,
                                                    cv_fold = args.cv_fold_num)

    return data