#include <stdlib.h>
#include "darknet.h"
#include "network.h"
#include "parser.h"
#include "box.h"
#include "option_list.h"

void train_detector_with_sensitivity(char *datacfg, char *cfgfile, char *weightfile, int clear)
{
    list *options = read_data_cfg(datacfg);
    char *train_images = option_find_str(options, "train", "data/train.txt");
    char *backup_directory = option_find_str(options, "backup", "/backup");
    char *train_labels = option_find_str(options, "train_labels", "data/train.labels");

    char *base = basecfg(cfgfile);
    printf("%s\n", base);

    // allocate networks
    network net = parse_network_cfg(cfgfile);
    load_weights(&net, weightfile);

    if (clear) {
        *net.seen = 0;
        *net.cur_iteration = 0;
    }

    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);

    int k_max = 0;
    for (int k = 0; k < net.n; ++k) {
        layer lk = net.layers[k];
        if (lk.type == YOLO || lk.type == GAUSSIAN_YOLO || lk.type == REGION) {
            k_max = k;
        }
    }

    for (int k = 0; k <= k_max; ++k) {
#ifdef GPU
        net.layers[k].update_gpu = 0;
#else
        net.layers[k].update = 0;
#endif
    }

    const int init_w = net.w;
    const int init_h = net.h;

    const int iter_log = 500, iter_save = 10000, iter_resize = 100;

    int dim_w = init_w;
    int dim_h = init_h;

    list *plist = get_paths(train_images);
    int train_images_num = plist->size;
    char **paths = (char **)list_to_array(plist);

    matrix labels = csv_to_matrix(train_labels);
    printf("labels => rows: %d, cols: %d\n", labels.rows, labels.cols);

    data train = {0};
    train.shallow = 0;

    load_args args = { 0 };
    args.w = net.w;
    args.h = net.h;
    args.c = net.c;
    args.type = IMAGE_DATA;

    int b = 0, t = 0;
    int nthreads = 4;
    if (nthreads > net.batch) {
        nthreads = net.batch;
    }

    int *im_ind = (int*)xcalloc(net.batch, sizeof(int));
    image* buf = (image*)xcalloc(net.batch, sizeof(image));
    image* buf_resized = (image*)xcalloc(net.batch, sizeof(image));
    pthread_t *thr = (pthread_t*)xcalloc(nthreads, sizeof(pthread_t));

    srand(8000);

    float total_loss = 0;
    char buff[512];

    double start_time = what_time_is_it_now(), cur_time, remain_time;
    double latency = 0;

    while (get_current_iteration(net) < net.max_batches) {
        const int iteration = get_current_iteration(net);

        // load images
        for (b = 0; b < net.batch; b += nthreads) {
            for (t = 0; t < nthreads && b + t < net.batch; ++t) {
                im_ind[b + t] = rand_int(0, train_images_num - 1);
                args.path = paths[im_ind[b + t]];
                args.im = &buf[b + t];
                args.resized = &buf_resized[b + t];
                thr[t] = load_data_in_thread(args);
            }

            for (t = 0; t < nthreads && b + t < net.batch; ++t) {
                pthread_join(thr[t], 0);
            }
        }

        // set up train data
        train.X.rows = net.batch;
        train.X.vals = (float**) xcalloc(net.batch, sizeof(float*));
        train.X.cols = buf_resized[0].w * buf_resized[0].h * buf_resized[0].c;

        train.y.rows = net.batch;
        train.y.vals = (float**) xcalloc(net.batch, sizeof(float*));
        train.y.cols = labels.cols;

        for (b = 0; b < net.batch; ++b) {
            train.X.vals[b] = buf_resized[b].data;
            train.y.vals[b] = labels.vals[im_ind[b]];
        }

        train_network(net, train);
        total_loss += net.layers[net.n -1].cost[0] / net.batch;

        for (b = 0; b < nthreads; ++b) {
            free_image(buf[b]);
            free_image(buf_resized[b]);
        }

        if (iteration % iter_log == 0) {
            cur_time = what_time_is_it_now();
            latency = (cur_time - start_time) / (iteration + 1);
            remain_time = (net.max_batches - iteration) * latency / 60 / 60;

            fprintf(stderr, "[%d] total loss: %f, avg loss: %f, learning rate: %f, %lf hours left\n", iteration, total_loss, total_loss / iter_log, get_current_rate(net), remain_time);
            total_loss = 0;
        }

        if (iteration % iter_save == 0) {
            resize_network(&net, init_w, init_h);
            sprintf(buff, "%s/%s_%d.weights", backup_directory, base, iteration);
            save_weights(net, buff);
        }

        if (iteration % iter_resize == 0) {
            // resize image every 100 iterations
            float random_val = rand_uniform_strong(0.5, 1.0);

            dim_w = roundl(random_val*init_w / net.resize_step + 1) * net.resize_step;
            dim_h = roundl(random_val*init_h / net.resize_step + 1) * net.resize_step;
            if (random_val < 1 && (dim_w > init_w || dim_h > init_h)) dim_w = init_w, dim_h = init_h;

            if (dim_w < net.resize_step) dim_w = net.resize_step;
            if (dim_h < net.resize_step) dim_h = net.resize_step;

            args.w = dim_w;
            args.h = dim_h;

            resize_network(&net, dim_w, dim_h);
        }

        free(train.X.vals);
        free(train.y.vals);
    }

    resize_network(&net, init_w, init_h);
    sprintf(buff, "%s/%s_final.weights", backup_directory, base);
    save_weights(net, buff);
    printf("If you want to train from the beginning, then use flag in the end of training command: -clear \n");

    free_matrix(labels);
    free_network(net);

    free(im_ind);
    free(buf);
    free(buf_resized);
    free(thr);
}

typedef struct {
    box b;
    float p;
    int class_id;
    int image_index;
    int truth_flag;
    int unique_truth_index;
} box_prob;

extern int detections_comparator(const void *pa, const void *pb);

typedef struct schedule {
    struct qentry *queue;
    float makespan;
} schedule;

typedef struct qentry {
    int stream_id;
    float sensitivity;
    int scale;
    int proc;
    float *track;
} qentry;

int qentry_comparator(const void *a, const void *b) {
    qentry *pa = (qentry *) a;
    qentry *pb = (qentry *) b;

    if(pa->sensitivity > pb->sensitivity) {
        return -1;
    }
    else if (pa->sensitivity < pb->sensitivity) {
        return 1;
    } else {
        return 0;
    }
}

float expected_accuracy_loss(const int scale_ind, const float sensitivity, const int nscales) {
    if (scale_ind == nscales - 1) {
        return (float) INT_MAX;
    }

    return pow(sensitivity, ((float)(scale_ind + 1))/((nscales - 1)));
}

void update_schedule(schedule *sc, const float *sensitivity, const float deadline, const matrix scales, const int nstreams, const int ngpus) {
    // initialize
    for (int k = 0; k < nstreams; k++) {
        sc->queue[k].stream_id = k;
        sc->queue[k].sensitivity = sensitivity[k];
        sc->queue[k].scale = 0;
    }
    qsort(sc->queue, nstreams, sizeof(qentry), qentry_comparator);

    float *w_p = (float*) xcalloc(ngpus, sizeof(float));
    for (int p = 0; p < ngpus; p++) {
        w_p[p] = 0;
    }

    for (int k = 0; k < nstreams; k++) {
        int p_star = k % ngpus;
        sc->queue[k].proc = p_star;

        for (int p = 0; p < ngpus; p++) {
            sc->queue[k].track[p] = w_p[p];
        }

        w_p[p_star] += scales.vals[0][2];
    }
    sc->makespan = w_p[max_index(w_p, ngpus)];

    // main loop
    float *val = (float*) xcalloc(nstreams, sizeof(float));
    while(sc->makespan > deadline) {
        for (int k = 0; k < nstreams; k++) {
            val[k] = expected_accuracy_loss(sc->queue[k].scale, sc->queue[k].sensitivity, scales.rows);
        }

        int k_min = min_index(val, nstreams);
        sc->queue[k_min].scale += 1;

        // update
        for (int p = 0; p < ngpus; p++) {
            w_p[p] = sc->queue[k_min].track[p];
        }
        w_p[sc->queue[k_min].proc] += scales.vals[sc->queue[k_min].scale][2];

        for (int k = k_min + 1; k < nstreams; k++) {
            int p_star = min_index(w_p, ngpus);
            sc->queue[k].proc = p_star;

            for (int p = 0; p < ngpus; p++) {
                sc->queue[k].track[p] = w_p[p];
            }

            w_p[p_star] += scales.vals[sc->queue[k].scale][2];
        }
        sc->makespan = w_p[max_index(w_p, ngpus)];
    }

    // optimize
    int changed = 1;
    int max_scale = 0;

    while(changed) {
        changed = 0;
        max_scale = 0;

        for (int k = 0; k < nstreams; k++) {
            if (sc->queue[k].scale == max_scale)
                continue;

            int old = scales.vals[sc->queue[k].scale][2];
            int new = scales.vals[sc->queue[k].scale - 1][2];

            if (w_p[sc->queue[k].proc] - old + new <= deadline) {
                sc->queue[k].scale -= 1;
                w_p[sc->queue[k].proc] += new - old;
                changed = 1;
            }

            if (k < nstreams - 1 && sc->queue[k].sensitivity != sc->queue[k].sensitivity) {
                max_scale = sc->queue[k].scale;
            }
        }
    }

    free(w_p);
    free(val);
}

void free_schedule(schedule sc, const int nstreams) {
    for (int k = 0; k < nstreams; k++) {
        free(sc.queue[k].track);
    }
    free(sc.queue);
}

network create_child_network(network base_net) {
    network child_net = base_net;

    // update the layers
    child_net.layers = (layer*) xcalloc(base_net.n, sizeof(layer));
    for (int i = 0; i < base_net.n; ++i) {
        layer l = base_net.layers[i];

        if (l.type == ROUTE || l.type == SHORTCUT) {
            l.input_sizes = (int*) xcalloc(base_net.layers[i].n, sizeof(int));
        }

#ifdef GPU
#ifdef CUDNN
        if (l.type == CONVOLUTIONAL) {
            create_convolutional_cudnn_tensors(&l);
        } else if (l.type == MAXPOOL) {
            create_maxpool_cudnn_tensors(&l);
        } else if (l.type == SHORTCUT) {
            l.input_sizes_gpu = cuda_make_int_array_new_api(l.input_sizes, l.n);
        }
#endif
#endif
        child_net.layers[i] = l;
    }

    return child_net;
}

void free_child_network(network net) {
    free(net.layers);
}

double validate_detector_with_sensitivity_map(char *datacfg, char *cfgfile, char *weightfile, float thresh_calc_avg_iou, const float iou_thresh, const int map_points, const float norm_min, const float norm_max, const int nstreams, const int ngpus, float deadline, char *dynamic_deadline) {
    list *options = read_data_cfg(datacfg);
    char *valid_images = option_find_str(options, "valid", "data/train.txt");
    char *difficult_valid_images = option_find_str(options, "difficult", NULL);
    char *name_list = option_find_str(options, "names", "data/names.list");
    int names_size = 0;
    char **names = get_labels_custom(name_list, &names_size); //get_labels(name_list);

    char *valid_labels = option_find_str(options, "valid_labels", "data/valid.labels");
    char *scale_list = option_find_str(options, "scale_names", "data/scale.names");

    matrix scales = csv_to_matrix(scale_list);
    matrix labels = csv_to_matrix(valid_labels);
    matrix deadlines;

    if (dynamic_deadline) {
        deadlines = csv_to_matrix(dynamic_deadline);
    }

    //char *mapf = option_find_str(options, "map", 0);
    //int *map = 0;
    //if (mapf) map = read_map(mapf);
    FILE* reinforcement_fd = NULL;

    network *base_nets = (network*) xcalloc(ngpus, sizeof(network));
    for (int p = 0; p < ngpus; p++) {
        cuda_set_device(p);

        base_nets[p] = parse_network_cfg_custom(cfgfile, 1, 1);    // set batch=1
        disable_memory_realloc(&base_nets[p]);

        load_weights(&base_nets[p], weightfile);

        fuse_conv_batchnorm(base_nets[p]);
        calculate_binary_weights(base_nets[p]);
    }

    network **nets = (network**) xcalloc(ngpus, sizeof(network*));
    for (int p = 0; p < ngpus; p++) {
        cuda_set_device(p);

        nets[p] = (network*) xcalloc(scales.rows, sizeof(network));
        for (int k = 0; k < scales.rows; k++) {
            // copy the baseline network
            nets[p][k] = create_child_network(base_nets[p]);

            // resize network for each scale
            resize_network(&nets[p][k], (int) scales.vals[k][0], (int) scales.vals[k][1]);
        }
    }

    srand(time(0));
    printf("\n calculation mAP (mean average precision)...\n");

    list *plist = get_paths(valid_images);
    char **paths = (char **)list_to_array(plist);

    list *plist_dif = NULL;
    char **paths_dif = NULL;
    if (difficult_valid_images) {
        plist_dif = get_paths(difficult_valid_images);
        paths_dif = (char **)list_to_array(plist_dif);
    }

    network ref_net = nets[0][0];
    layer l = ref_net.layers[ref_net.n - 1];
    int k, k_max;
    for (k = 0; k < ref_net.n; ++k) {
        layer lk = ref_net.layers[k];
        if (lk.type == YOLO || lk.type == GAUSSIAN_YOLO || lk.type == REGION) {
            l = lk;
            k_max = k;
            printf(" Detection layer: %d - type = %d \n", k, l.type);
        }
    }
    int classes = l.classes;

    if (classes != names_size) {
        printf("\n Error: in the file %s number of names %d that isn't equal to classes=%d in the file %s \n",
            name_list, names_size, classes, cfgfile);
        getchar();
    }

    int m = plist->size;
    int index = 0;
    int t;

    const float thresh = .005;
    const float nms = .45;
    //const float iou_thresh = 0.5;

    image* buf = (image*)xcalloc(nstreams, sizeof(image));
    image* buf_resized = (image*)xcalloc(nstreams, sizeof(image));
    pthread_t* thr = (pthread_t*)xcalloc(nstreams, sizeof(pthread_t));

    // scale_ind indicates the scales for the current images
    int proc_ind = 0;
    int scale_ind = 0;

    float *sensitivity = (float*) xcalloc(nstreams, sizeof(float));

    load_args args = { 0 };
    args.w = ref_net.w;
    args.h = ref_net.h;
    args.c = ref_net.c;
    args.type = IMAGE_DATA;

    //const float thresh_calc_avg_iou = 0.24;
    float avg_iou = 0;
    int tp_for_thresh = 0;
    int fp_for_thresh = 0;

    box_prob* detections = (box_prob*)xcalloc(1, sizeof(box_prob));
    int detections_count = 0;
    int unique_truth_count = 0;

    int* truth_classes_count = (int*)xcalloc(classes, sizeof(int));

    // For multi-class precision and recall computation
    float *avg_iou_per_class = (float*)xcalloc(classes, sizeof(float));
    int *tp_for_thresh_per_class = (int*)xcalloc(classes, sizeof(int));
    int *fp_for_thresh_per_class = (int*)xcalloc(classes, sizeof(int));

    time_t start = time(0);

    struct schedule sc;
    sc.queue = (struct qentry*) xcalloc(nstreams, sizeof(struct qentry));
    for (int k = 0; k < nstreams; k++) {
        sc.queue[k].track = (float*)xcalloc(ngpus, sizeof(float));
        for (int p = 0; p < ngpus; p++) {
            sc.queue[k].track[p] = 0;
        }
    }

    int frame_id = 0, count = 0;

    for (index = 0; index < m; index += nstreams) {
        frame_id = index / nstreams;

        if (dynamic_deadline) {
            deadline = deadlines.vals[frame_id][0];
        }

        update_schedule(&sc, sensitivity, deadline, scales, nstreams, ngpus);

        for (count = 0; count < nstreams && index + count < m; ++count) {
            // get next index
            t = sc.queue[count].stream_id;

            proc_ind = sc.queue[count].proc;
            scale_ind = sc.queue[count].scale;

            // load image
            args.path = paths[index + t];
            args.im = &buf[t];
            args.resized = &buf_resized[t];
            args.w = (int) scales.vals[scale_ind][0];
            args.h = (int) scales.vals[scale_ind][1];

            thr[t] = load_data_in_thread(args);
            pthread_join(thr[t], 0);

            const int image_index = index + t;
            char *path = paths[image_index];
            char *id = basecfg(path);
            float *X = buf_resized[t].data;

            cuda_set_device(proc_ind);
            sensitivity[t] = network_predict(nets[proc_ind][scale_ind], X)[0];

            // denormalize sensitivity
            sensitivity[t] = sensitivity[t] * (norm_max - norm_min) + norm_min;

            int nboxes = 0;
            float hier_thresh = 0;
            detection *dets;
            dets = get_network_boxes_with_index(&nets[proc_ind][scale_ind], 1, 1, thresh, hier_thresh, 0, 0, &nboxes, 0, k_max);

            if (nms) {
                if (l.nms_kind == DEFAULT_NMS) do_nms_sort(dets, nboxes, l.classes, nms);
                else diounms_sort(dets, nboxes, l.classes, nms, l.nms_kind, l.beta_nms);
            }

            char labelpath[4096];
            replace_image_to_label(path, labelpath);
            int num_labels = 0;
            box_label *truth = read_boxes(labelpath, &num_labels);
            int j;
            for (j = 0; j < num_labels; ++j) {
                truth_classes_count[truth[j].id]++;
            }

            // difficult
            box_label *truth_dif = NULL;
            int num_labels_dif = 0;
            if (paths_dif)
            {
                char *path_dif = paths_dif[image_index];

                char labelpath_dif[4096];
                replace_image_to_label(path_dif, labelpath_dif);

                truth_dif = read_boxes(labelpath_dif, &num_labels_dif);
            }

            const int checkpoint_detections_count = detections_count;

            int i;
            for (i = 0; i < nboxes; ++i) {

                int class_id;
                for (class_id = 0; class_id < classes; ++class_id) {
                    float prob = dets[i].prob[class_id];
                    if (prob > 0) {
                        detections_count++;
                        detections = (box_prob*)xrealloc(detections, detections_count * sizeof(box_prob));
                        detections[detections_count - 1].b = dets[i].bbox;
                        detections[detections_count - 1].p = prob;
                        detections[detections_count - 1].image_index = image_index;
                        detections[detections_count - 1].class_id = class_id;
                        detections[detections_count - 1].truth_flag = 0;
                        detections[detections_count - 1].unique_truth_index = -1;

                        int truth_index = -1;
                        float max_iou = 0;
                        for (j = 0; j < num_labels; ++j)
                        {
                            box t = { truth[j].x, truth[j].y, truth[j].w, truth[j].h };
                            //printf(" IoU = %f, prob = %f, class_id = %d, truth[j].id = %d \n",
                            //    box_iou(dets[i].bbox, t), prob, class_id, truth[j].id);
                            float current_iou = box_iou(dets[i].bbox, t);
                            if (current_iou > iou_thresh && class_id == truth[j].id) {
                                if (current_iou > max_iou) {
                                    max_iou = current_iou;
                                    truth_index = unique_truth_count + j;
                                }
                            }
                        }

                        // best IoU
                        if (truth_index > -1) {
                            detections[detections_count - 1].truth_flag = 1;
                            detections[detections_count - 1].unique_truth_index = truth_index;
                        }
                        else {
                            // if object is difficult then remove detection
                            for (j = 0; j < num_labels_dif; ++j) {
                                box t = { truth_dif[j].x, truth_dif[j].y, truth_dif[j].w, truth_dif[j].h };
                                float current_iou = box_iou(dets[i].bbox, t);
                                if (current_iou > iou_thresh && class_id == truth_dif[j].id) {
                                    --detections_count;
                                    break;
                                }
                            }
                        }

                        // calc avg IoU, true-positives, false-positives for required Threshold
                        if (prob > thresh_calc_avg_iou) {
                            int z, found = 0;
                            for (z = checkpoint_detections_count; z < detections_count - 1; ++z) {
                                if (detections[z].unique_truth_index == truth_index) {
                                    found = 1; break;
                                }
                            }

                            if (truth_index > -1 && found == 0) {
                                avg_iou += max_iou;
                                ++tp_for_thresh;
                                avg_iou_per_class[class_id] += max_iou;
                                tp_for_thresh_per_class[class_id]++;
                            }
                            else{
                                fp_for_thresh++;
                                fp_for_thresh_per_class[class_id]++;
                            }
                        }
                    }
                }
            }

            unique_truth_count += num_labels;

            //static int previous_errors = 0;
            //int total_errors = fp_for_thresh + (unique_truth_count - tp_for_thresh);
            //int errors_in_this_image = total_errors - previous_errors;
            //previous_errors = total_errors;
            //if(reinforcement_fd == NULL) reinforcement_fd = fopen("reinforcement.txt", "wb");
            //char buff[1000];
            //sprintf(buff, "%s\n", path);
            //if(errors_in_this_image > 0) fwrite(buff, sizeof(char), strlen(buff), reinforcement_fd);

            free_detections(dets, nboxes);
            free(truth);
            free(truth_dif);
            free(id);
            free_image(buf[t]);
            free_image(buf_resized[t]);
        }
    }

    if ((tp_for_thresh + fp_for_thresh) > 0)
        avg_iou = avg_iou / (tp_for_thresh + fp_for_thresh);

    int class_id;
    for(class_id = 0; class_id < classes; class_id++){
        if ((tp_for_thresh_per_class[class_id] + fp_for_thresh_per_class[class_id]) > 0)
            avg_iou_per_class[class_id] = avg_iou_per_class[class_id] / (tp_for_thresh_per_class[class_id] + fp_for_thresh_per_class[class_id]);
    }

    // SORT(detections)
    qsort(detections, detections_count, sizeof(box_prob), detections_comparator);

    typedef struct {
        double prob;
        double precision;
        double recall;
        int tp, fp, fn;
    } pr_t;

    // for PR-curve
    pr_t** pr = (pr_t**)xcalloc(classes, sizeof(pr_t*));
    for (int i = 0; i < classes; ++i) {
        pr[i] = (pr_t*)xcalloc(detections_count, sizeof(pr_t));
    }
    printf("\n detections_count = %d, unique_truth_count = %d  \n", detections_count, unique_truth_count);


    int* detection_per_class_count = (int*)xcalloc(classes, sizeof(int));
    for (int j = 0; j < detections_count; ++j) {
        detection_per_class_count[detections[j].class_id]++;
    }

    int* truth_flags = (int*)xcalloc(unique_truth_count, sizeof(int));

    int rank;
    for (rank = 0; rank < detections_count; ++rank) {
        if (rank % 100 == 0)
            printf(" rank = %d of ranks = %d \r", rank, detections_count);

        if (rank > 0) {
            int class_id;
            for (class_id = 0; class_id < classes; ++class_id) {
                pr[class_id][rank].tp = pr[class_id][rank - 1].tp;
                pr[class_id][rank].fp = pr[class_id][rank - 1].fp;
            }
        }

        box_prob d = detections[rank];
        pr[d.class_id][rank].prob = d.p;
        // if (detected && isn't detected before)
        if (d.truth_flag == 1) {
            if (truth_flags[d.unique_truth_index] == 0)
            {
                truth_flags[d.unique_truth_index] = 1;
                pr[d.class_id][rank].tp++;    // true-positive
            } else
                pr[d.class_id][rank].fp++;
        }
        else {
            pr[d.class_id][rank].fp++;    // false-positive
        }

        for (int i = 0; i < classes; ++i)
        {
            const int tp = pr[i][rank].tp;
            const int fp = pr[i][rank].fp;
            const int fn = truth_classes_count[i] - tp;    // false-negative = objects - true-positive
            pr[i][rank].fn = fn;

            if ((tp + fp) > 0) pr[i][rank].precision = (double)tp / (double)(tp + fp);
            else pr[i][rank].precision = 0;

            if ((tp + fn) > 0) pr[i][rank].recall = (double)tp / (double)(tp + fn);
            else pr[i][rank].recall = 0;

            if (rank == (detections_count - 1) && detection_per_class_count[i] != (tp + fp)) {    // check for last rank
                    printf(" class_id: %d - detections = %d, tp+fp = %d, tp = %d, fp = %d \n", i, detection_per_class_count[i], tp+fp, tp, fp);
            }
        }
    }

    free(truth_flags);

    double mean_average_precision = 0;
    int valid_classes = 0;

    for (int i = 0; i < classes; ++i) {
        double avg_precision = 0;

        // MS COCO - uses 101-Recall-points on PR-chart.
        // PascalVOC2007 - uses 11-Recall-points on PR-chart.
        // PascalVOC2010-2012 - uses Area-Under-Curve on PR-chart.
        // ImageNet - uses Area-Under-Curve on PR-chart.

        // correct mAP calculation: ImageNet, PascalVOC 2010-2012
        if (map_points == 0)
        {
            double last_recall = pr[i][detections_count - 1].recall;
            double last_precision = pr[i][detections_count - 1].precision;
            for (rank = detections_count - 2; rank >= 0; --rank)
            {
                double delta_recall = last_recall - pr[i][rank].recall;
                last_recall = pr[i][rank].recall;

                if (pr[i][rank].precision > last_precision) {
                    last_precision = pr[i][rank].precision;
                }

                avg_precision += delta_recall * last_precision;
            }
            //add remaining area of PR curve when recall isn't 0 at rank-1
            double delta_recall = last_recall - 0;
            avg_precision += delta_recall * last_precision;
        }
        // MSCOCO - 101 Recall-points, PascalVOC - 11 Recall-points
        else
        {
            int point;
            for (point = 0; point < map_points; ++point) {
                double cur_recall = point * 1.0 / (map_points-1);
                double cur_precision = 0;
                double cur_prob = 0;
                for (rank = 0; rank < detections_count; ++rank)
                {
                    if (pr[i][rank].recall >= cur_recall) {    // > or >=
                        if (pr[i][rank].precision > cur_precision) {
                            cur_precision = pr[i][rank].precision;
                            cur_prob = pr[i][rank].prob;
                        }
                    }
                }
                //printf("class_id = %d, point = %d, cur_prob = %.4f, cur_recall = %.4f, cur_precision = %.4f \n", i, point, cur_prob, cur_recall, cur_precision);

                avg_precision += cur_precision;
            }
            avg_precision = avg_precision / map_points;
        }

        printf("class_id = %d, name = %s, ap = %2.2f%%   \t (TP = %d, FP = %d) \n",
            i, names[i], avg_precision * 100, tp_for_thresh_per_class[i], fp_for_thresh_per_class[i]);

        //float class_precision = (float)tp_for_thresh_per_class[i] / ((float)tp_for_thresh_per_class[i] + (float)fp_for_thresh_per_class[i]);
        //float class_recall = (float)tp_for_thresh_per_class[i] / ((float)tp_for_thresh_per_class[i] + (float)(truth_classes_count[i] - tp_for_thresh_per_class[i]));
        //printf("Precision = %1.2f, Recall = %1.2f, avg IOU = %2.2f%% \n\n", class_precision, class_recall, avg_iou_per_class[i]);

        if (truth_classes_count[i] > 0) {
            mean_average_precision += avg_precision;
            valid_classes++;
        }
    }

    const float cur_precision = (float)tp_for_thresh / ((float)tp_for_thresh + (float)fp_for_thresh);
    const float cur_recall = (float)tp_for_thresh / ((float)tp_for_thresh + (float)(unique_truth_count - tp_for_thresh));
    const float f1_score = 2.F * cur_precision * cur_recall / (cur_precision + cur_recall);
    printf("\n for conf_thresh = %1.2f, precision = %1.2f, recall = %1.2f, F1-score = %1.2f \n",
        thresh_calc_avg_iou, cur_precision, cur_recall, f1_score);

    printf(" for conf_thresh = %0.2f, TP = %d, FP = %d, FN = %d, average IoU = %2.2f %% \n",
        thresh_calc_avg_iou, tp_for_thresh, fp_for_thresh, unique_truth_count - tp_for_thresh, avg_iou * 100);

    mean_average_precision = mean_average_precision / valid_classes;
    printf("\n IoU threshold = %2.0f %%, ", iou_thresh * 100);
    if (map_points) printf("used %d Recall-points \n", map_points);
    else printf("used Area-Under-Curve for each unique Recall \n");

    printf(" mean average precision (mAP@%0.2f) = %f, or %2.2f %% \n", iou_thresh, mean_average_precision, mean_average_precision * 100);

    for (int i = 0; i < classes; ++i) {
        free(pr[i]);
    }
    free(pr);
    free(detections);
    free(truth_classes_count);
    free(detection_per_class_count);
    free(paths);
    free(paths_dif);
    free_list_contents(plist);
    free_list(plist);
    if (plist_dif) {
        free_list_contents(plist_dif);
        free_list(plist_dif);
    }

    free(avg_iou_per_class);
    free(tp_for_thresh_per_class);
    free(fp_for_thresh_per_class);

    fprintf(stderr, "Total Detection Time: %d Seconds\n", (int)(time(0) - start));
    printf("\nSet -points flag:\n");
    printf(" `-points 101` for MS COCO \n");
    printf(" `-points 11` for PascalVOC 2007 (uncomment `difficult` in voc.data) \n");
    printf(" `-points 0` (AUC) for ImageNet, PascalVOC 2010-2012, your custom dataset\n");

    if (reinforcement_fd != NULL) fclose(reinforcement_fd);

    // free memory
    free_ptrs((void**)names, ref_net.layers[ref_net.n - 1].classes);
    free_list_contents_kvp(options);
    free_list(options);

    for (int p = 0; p < ngpus; p++) {
        for (int k = 0; k < scales.rows; k++) {
            free_child_network(nets[p][k]);
        }
        free(nets[p]);
        free_network(base_nets[p]);
    }
    free(nets);

    free_matrix(scales);
    free_matrix(labels);

    free(sensitivity);

    free_schedule(sc, nstreams);

    if (thr) free(thr);
    if (buf) free(buf);
    if (buf_resized) free(buf_resized);

    return mean_average_precision;
}

void run_detector_with_sensitivity(int argc, char **argv)
{
    float thresh = find_float_arg(argc, argv, "-thresh", .25);    // 0.24
    float iou_thresh = find_float_arg(argc, argv, "-iou_thresh", .5);    // 0.5 for mAP
    int map_points = find_int_arg(argc, argv, "-points", 0);
    int resume = find_arg(argc, argv, "-resume");

    float norm_min = find_float_arg(argc, argv, "-norm_min", .6);
    float norm_max = find_float_arg(argc, argv, "-norm_max", 2.8);

    float deadline = find_float_arg(argc, argv, "-deadline", 200);
    char *dynamic_deadline = find_char_arg(argc, argv, "-dynamic_deadline", 0);

    int nstreams = find_int_arg(argc, argv, "-nstreams", 3);
    int ngpus = find_int_arg(argc, argv, "-ngpus", 1);

    if (argc < 4) {
        fprintf(stderr, "usage: %s %s [train/test] [datacfg] [cfg] [weights]\n", argv[0], argv[1]);
        return;
    }

    int clear = find_arg(argc, argv, "-clear");

    char *datacfg = argv[3];
    char *cfg = argv[4];
    char *weights = argv[5];

    if (0 == strcmp(argv[2], "train")) train_detector_with_sensitivity(datacfg, cfg, weights, clear);
    else if (0 == strcmp(argv[2], "map")) validate_detector_with_sensitivity_map(datacfg, cfg, weights, thresh, iou_thresh, map_points, norm_min, norm_max, nstreams, ngpus, deadline, dynamic_deadline);

    else printf(" There isn't such command: %s", argv[2]);
}
