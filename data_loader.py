import data_placenta
import data_prostate

def load_data(args, train_test_val):

    if args.dataset == 'placenta':
        data_path = '/data/vision/polina/projects/fetal/projects/placenta-segmentation/data/split-nifti-processed/'
        images, labels, image_paths, label_paths = data_placenta.load_images_and_labels(data_path,
                                                                                        train_test_val = train_test_val,
                                                                                        cv_fold = args.cv_fold_num)

    elif args.dataset == 'prostate':
        data_path = '/data/vision/polina/users/nkarani/data/segmentation/prostate/'
        images, labels, image_paths, label_paths = data_prostate.load_images_and_labels(data_path,
                                                                                        args.sub_dataset,
                                                                                        train_test_val = train_test_val,
                                                                                        cv_fold = args.cv_fold_num)

    return images, labels, image_paths, label_paths