#include <opencv2/opencv.hpp>
#include "tensorflow/core/public/session.h"
#include "tensorflow/cc/ops/array_ops.h"
#include <dirent.h>
#include <Eigen/Dense>


//using namespace Eigen;
using namespace tensorflow;
using namespace std;
using namespace cv;



static const char* root_dataset = "../data/validation";
static string graphPath = "../models/graph_with_weights3.pb";

// ------ hyper parameters ------ //
static int __hp__response_up = 8;
static float __hp__window_influence = 0.25;
static float __hp__z_lr = 0.01;
static int __hp__scale_num = 3;
static double __hp__scale_step = 1.04;
static float __hp__scale_penalty = 0.97;
static float __hp__scale_lr = 0.59;
static float __hp__scale_min = 0.2;
static float __hp__scale_max = 5;

// ------ run conf ------ //
static float __run__visualization = 1;
static float __run__debug = 0;
static float __run__gpus = 1; // [1]

// ------ design params ------ //
static string __design__join_method = "xcorr";
static string __design__net = "baseline-conv5_e55.mat";
static string __design__net_gray = "";
static string __design__windowing = "cosine_sum";
static float __design__exemplar_sz = 127;
static float __design__search_sz = 255;
static int __design__score_sz = 33;
static int __design__tot_stride = 4;
static float __design__context = 0.5;
static bool __design__pad_with_image_mean = true;

// ------ eval params ------ //
static int __eval__n_subseq = 3;
static float __eval__dist_threshold = 20;
static float __eval__stop_on_failure = 0;
static string __eval__dataset = "validation";
static string __eval__video = "all";
static int __eval__start_frame = 0;


// --------- Layer names ----------------
static string input_name = "image_tensor:0";
static string boxes_name = "detection_boxes:0";
static string class_name = "detection_classes:0";
static string scores_name = "detection_scores:0";
static string detect_count_name = "num_detections:0";


Rect getBox(float ymin, float xmin, float ymax, float xmax, int imWidth ,int imHeight)
{
    int left = round(xmin * imWidth);
    int right = round(xmax * imWidth);
    int top = round(ymin * imHeight);
    int bottom = round(ymax * imHeight);
    Rect box(left, top, right-left, bottom-top);
    return box;
}


//void FillValues(Tensor* tensor, Mat vals, int size) {
//    auto flat = tensor->flat<uint8>();
//    if (flat.size() > 0) {
//        copy_n(vals.data, size, flat.data());
//    }
//}

int main(int argc, char* argv[])
{
    int final_score_sz = __hp__response_up * (__design__score_sz - 1) + 1;
    //cout << final_score_sz << endl;
    //cout << root_dataset << endl;

    // burda build graph var bu kisim lazim oldu simdi



    // # 1 - read directory vs
    cout << "###################### 1" << endl;
    vector<string> videos_list;
    DIR *dir = NULL;
    struct dirent *ent;
    if ((dir = opendir (root_dataset)) != NULL) {
        /* print all the files and directories within directory */
        while ((ent = readdir (dir)) != NULL) {
            if( strcmp(ent->d_name, ".") != 0 && strcmp(ent->d_name, "..") != 0 ){
                //printf ("%s\n", ent->d_name);
                string name = ent->d_name;
                videos_list.push_back(name);
            }

        }
        closedir (dir);
    } else {
        /* could not open directory */
        perror ("");
        return EXIT_FAILURE;
    }
    dir = NULL;
    sort(videos_list.begin(),videos_list.end());

//    for (auto const& c : videos_list)
//        cout << c << ' ';
//    cout << endl;

    int nv = videos_list.size();
    //cout << nv << endl;

    int speed[nv * __eval__n_subseq];
    memset(speed,0,sizeof(speed));
    //cout << speed[0] << endl;

    // # 2
    cout << "###################### 2" << endl;
    char buffer[256];
    for( int yy = 0; yy < nv; yy++ ) {
        strcpy(buffer, root_dataset);
        // # 3 gt frame_name_list frame_sz n_frames
        cout << "###################### 3" << endl;
        cout << "yy = " << yy << endl;
//        cout << "buffer (before) : " << buffer << endl;
//        cout << "videos_list[i] : " << videos_list[yy] << endl;

        strcat(buffer, "/");
        strcat(buffer, videos_list[yy].c_str());

//        cout << "-->" << endl;
//        cout << "buffer (after) : " << buffer << endl;

        // >>>
        // frame name list okuma
        vector<string> frame_name_list;
        DIR *video_folder = NULL;
        if ((video_folder = opendir (buffer)) != NULL) {
            /* print all the files and directories within directory */
            while ((ent = readdir (video_folder)) != NULL) {
                if( strcmp(ent->d_name, ".") != 0 && strcmp(ent->d_name, "..") != 0 ){
                    string temp = ent->d_name;
                    if(!strcmp(&ent->d_name[temp.size()-4],".jpg")){
                        // cout << ent->d_name << ' ';

                        string frame_name = string(buffer) + "/" + string(ent->d_name);
                        // cout << frame_name << endl;
                        frame_name_list.push_back(frame_name);
                    }
                }
            }
            closedir (dir);
        } else {
            /* could not open directory */
            perror ("");
            return EXIT_FAILURE;
        }

        sort(frame_name_list.begin(),frame_name_list.end());
//        for (auto const& c : frame_name_list)
//            cout << c << ' ';
        // END


        // >>>
        // frame size okuma
        Mat image;
        image = imread(frame_name_list[0], CV_LOAD_IMAGE_COLOR);

//        cout << endl;

        vector<int> frame_sz;
        frame_sz.push_back(image.size().width);
        frame_sz.push_back(image.size().height);

//        for (auto const& c : frame_sz)
//            cout << c << endl;
//        // END

        // ground truth okuma
        string gt_name = "groundtruth.txt";
        string gt_file = string(buffer) + string("/") + string(gt_name);
        //cout << "gt_file : " << gt_file << endl;

        ifstream in(gt_file);

        char str[255];
        vector<vector<string>> gt{{}};
        while(in){
            in.getline(str,255);
            if(in){
                //cout << str << endl;
                gt[0].push_back((string)str);
            }
        }

        in.close();

//        for (int i = 0; i < gt.size(); i++)
//        {
//            for (int j = 0; j < gt[i].size(); j++)
//            {
//                cout << gt[i][j];
//            }
//        }
        // END

        // !! gt, frame_name_list, frame_sz, n_frames available here

        // # 4
        cout << "###################### 4" << endl;
        //cout << "frame name list size : " << frame_name_list.size() << endl;


        // TODO burda start frameler birebir degil +-1 seklinde gozlemleniyor düzeltme gerekebilir
        float num = __eval__n_subseq + 1;
        float stop = frame_name_list.size();
        float start = 0;
        float delta = stop - start;
        float div = num - 1;

        float step = delta / div;
        //cout << "step : " << step << endl;
        //cout << num << endl;
        vector<float> starts;
        for (int j=0; j < num ; j++){
            float &&start = floor(step) * j;
            cout << " " << start << endl; // start frameleri basiyor
            starts.push_back(int(start));
        }

//        cout << i << endl;
//        cout << "wtf" << endl;

        // slice job
        vector<int> starts_;
        for (int k = 0; k < __eval__n_subseq; ++k) {
            starts_.push_back(int(starts[k]));
        }

//        for (int l = 0; l < starts_.size(); ++l) {
//            cout << starts_[l] << endl;
//        }
        // END

        // Trackerin calisacagi subseqler icin fora giriyoruz
        // # 5
        cout << "###################### 5" << endl;
        for (int zz = 0; zz < __eval__n_subseq; zz++) {
            // # 6
            cout << "###################### 6" << endl;
            cout << "z = " << zz << endl;
            int start_frame = starts_[zz];
            cout << "start frame : " << start_frame << endl;

            // ilgili ground trutch parcacigini ayiklama {gt_}
            vector<vector<string>> gt_{{}};
            for (int k = start_frame; k < gt[0].size(); ++k) {
                gt_[0].push_back((string)gt[0][k]);
            }
//            for (int l = 0; l < gt_[0].size(); ++l) {
//                cout << gt_[0][l] << endl;
//            }

            // ilgili frame nameler icin ayiklama = slice {frame_name_list_}
            vector<string> frame_name_list_;
            for (int k = start_frame; k < frame_name_list.size(); ++k) {
                frame_name_list_.push_back(frame_name_list[k]);
            }
//            for (int l = 0; l < frame_name_list_.size(); ++l) {
//                cout << frame_name_list_[l] << endl;
//            }


            // bu kisim region to bbox fonksiyonu
            double pos_x, pos_y, target_w, target_h;
            //cout << gt_[0][0] << endl;

            string s = gt_[0][0];

            string delimiter = ",";

            size_t pos = 0;
            string token;

            vector<int> region;
            while ((pos = s.find(delimiter)) != string::npos) {
                token = s.substr(0, pos);
                //  cout << "token : " << token << endl;
                region.push_back(stoi(token));
                s.erase(0, pos + delimiter.length());
            }
            //cout << s << endl;
            region.push_back(stoi(s));

//            for (int l = 0; l < region.size(); ++l) {
//                cout << region[l] << endl;
//            }

            float x = region[0];
            float y = region[1];
            float w = region[2];
            float h = region[3];

            float cx = x+w/2;
            float cy = y+h/2;

            pos_x = cx;
            pos_y = cy;
            target_w = w;
            target_h = h;

//            cout << "parameters" << endl;
//            cout << pos_x << endl;
//            cout << pos_y << endl;
//            cout << target_w << endl;
//            cout << target_h << endl;




//            cout << " # # # # # " << endl;
//            cout << y << endl;
//            cout << z << endl;
//            cout << n_subseq << endl;
//            cout << " # # # # # " << endl;

            int idx = yy * __eval__n_subseq + zz;
            //cout << "idx : " << idx << endl;

            // # 7 ONEMLIII !! asil tracker basliyor
            //
            // Parametreler:
            //              hp, run, design, frame_name_list_, pos_x, pos_y,target_w, target_h, final_score_sz,
            //              filename, image, templates_z, scores, start_frame
            // TRACKER IMPLEMENTATION (def tracker)

            // # 7
            cout << "###################### 7" << endl;
            int num_frames = frame_name_list_.size();
            cout << "=============>>>>" << num_frames << endl;

            // # 8
            cout << "###################### 8" << endl;
            Eigen::ArrayXXf bboxes = Eigen::ArrayXXf::Zero(num_frames,4);
            //cout << bboxes << endl;

            // # 9
            cout << "###################### 9" << endl;

            // TODO (SOLVED) int olacakmis python tarafinde -1 0 1 gelen pre_scale burda -2 0 2 geliyor
//            cout << __hp__scale_num / 2;
//            cout << ceil(__hp__scale_num / 2 );

            Eigen::VectorXd pre_scale_factors = Eigen::VectorXd::LinSpaced(__hp__scale_num ,-ceil(__hp__scale_num / 2),ceil(__hp__scale_num / 2));

            //cout << pre_scale_factors(0) << endl;

            vector<double> scale_factors(pre_scale_factors.data(), pre_scale_factors.data() + pre_scale_factors.rows()*pre_scale_factors.cols());
            //cout << scale_factors[0] << endl;
            //cout << scale_factors[1] << endl;
            //cout << scale_factors[2] << endl;

            for (int m = 0; m < scale_factors.size(); ++m) {
                scale_factors[m] = pow(__hp__scale_step,scale_factors[m]);
            }

            //cout << scale_factors[0] << endl;
            //cout << scale_factors[1] << endl;
            //cout << scale_factors[2] << endl;

            // TODO (SOLVED) yukarida halloldu vectore atip islem yapinca : scale factors yarim kaldi power alinamadi


            // # 10
            cout << "###################### 10" << endl;

            // np.hanning implementation n = arrange(M)
            Eigen::MatrixXd n(1,final_score_sz);
            for (int i = 0; i < final_score_sz; ++i) {
                n(0,i) = i;
            }
            //cout << n.rows() << n.cols() << endl;

            // Gerek yok
//            Tensor hann_1d(DT_DOUBLE,{257,257});
//            auto hann_1d_mapped = hann_1d.tensor<double, 2>();
//            for (int i = 0; i < final_score_sz; ++i) {
//                n(0,i) = 0.5 - 0.5*cos(2*M_1_PI*i/(final_score_sz-1));
//            }


            // np.arange(0,257)

            for (int i = 0; i < final_score_sz; ++i) {
                n(0,i) = i;
            }

//            for (int i = 0; i < final_score_sz; ++i) {
//                cout << n(0,i) << " ";
//            }

            cout << endl;
            double pi = 3.141592653589793;
            for (int i = 0; i < final_score_sz; ++i) {
                n(0,i) = 0.5 - 0.5*cos(2*pi*n(0,i)/(final_score_sz-1));
            }

//            cout << "hann_1d : (0,1) = " << n(0,1) << endl;
//            cout << "hann_1d : (0,2) = " << n(0,2) << endl;
//            cout << "hann_1d : (0,3) = " << n(0,3) << endl;
//            cout << "hann_1d : (0,4) = " << n(0,4) << endl;
//            cout << "hann_1d : (0,5) = " << n(0,5) << endl;
//            cout << "hann_1d : (0,145) = " << n(0,145) << endl;
//            cout << "hann_1d : (0,255) = " << n(0,255) << endl;
//            cout << n.rows() << " x " << n.cols() << endl;
//            cout << n.size() << endl;
            // TODO burda hann_1d buyuk olcude tamam sadece expand dims tutmuyor olabilir
            // TODO sayilar birebir ayni degil benzer sekilde bu neden bi arastirmak lazim
            // VectorXd hann_1d =


            //a = asarray(a)
            //shape = a.shape

            //return a.reshape(shape[:axis] + (1,) + shape[axis:])



            // # 11
            cout << "###################### 11" << endl;

            Eigen::MatrixXd penalty;
            penalty = n.transpose() * n;
            //cout << penalty.rows() << " x " << penalty.cols() << endl;

            //penalty = np.transpose(hann_1d) * hann_1d
            //penalty = penalty / np.sum(penalty)


            // # 12
            cout << "###################### 12" << endl;
            //cout << "design.context : " << __design__context << endl;
            //cout << "target w : " << target_w << endl;
            //cout << "target_h : " << target_h << endl;
            float context = __design__context*(target_w+target_h);
            //cout << context << endl;

            double z_sz = sqrt((target_w+context)*(target_h+context));
            //cout << z_sz << endl;

            double x_sz = __design__search_sz / __design__exemplar_sz * z_sz;
            //cout << x_sz << endl;
            //context = design.context*(target_w+target_h)
            //print "__________________------------->>>>>"
            //print type(context)
            //z_sz = np.sqrt(np.prod((target_w+context)*(target_h+context)))
            //x_sz = float(design.search_sz) / design.exemplar_sz * z_sz


            // # 13
            cout << "###################### 13" << endl;

            float min_z = __hp__scale_min * z_sz;
            float max_z = __hp__scale_max * z_sz;
            float min_x = __hp__scale_min * x_sz;
            float max_x = __hp__scale_max * x_sz;

            //cout << "min_z : " << min_z << endl;
            //cout << "max_z : " << max_z << endl;
            //cout << "min_x : " << min_x << endl;
            //cout << "max_x : " << max_x << endl;

            // # 14 session !!!
            cout << "###################### 14 session is below" << endl;

            // save first frame position (from ground-truth)
            bboxes(0,0) = pos_x-target_w/2;
            bboxes(0,1) = pos_y-target_h/2;
            bboxes(0,2) = target_w;
            bboxes(0,3) = target_h;
            //cout << bboxes << endl;


            // # 15
            cout << "###################### 15" << endl;
            Session* session1;
            Status status = NewSession(SessionOptions(),&session1);
            if (!status.ok())
            {
                cout << status.ToString() << "\n";
                return 1;
            }

            GraphDef graph_def;
            status = ReadBinaryProto(Env::Default(), graphPath, &graph_def);
            if (!status.ok())
            {
                cout << status.ToString() << "\n";
                return 1;
            }

            status = session1->Create(graph_def);
            if (!status.ok())
            {
                cout << status.ToString() << "\n";
                return 1;
            }

            //Tensor siam__pos_y_ph(DT_DOUBLE, pos_y);
            //Tensor siam__z_sz_ph(DT_DOUBLE, z_sz);
            //Tensor filename(DT_STRING, frame_name_list_[0]);

            //cout << "A" << endl;

//            Tensor siam__pos_x_ph(DT_DOUBLE, {1});
            Tensor siam__pos_x_ph(DT_DOUBLE,TensorShape());
//            cout << "x" << endl;
//            auto siam__pos_x_ph_mapped = siam__pos_x_ph.tensor<double, 1>();
//            cout << "y" << endl;
            siam__pos_x_ph.scalar<double>()() = pos_x;
//            cout << "z" << endl;
//
//            cout << "B" << endl;

            Tensor siam__pos_y_ph(DT_DOUBLE,TensorShape());
//            auto siam__pos_y_ph_mapped = siam__pos_y_ph.tensor<double, 1>();
            siam__pos_y_ph.scalar<double>()() = pos_y;

            //cout << "C" << endl;

            Tensor siam__z_sz_ph(DT_DOUBLE,TensorShape());
//            auto siam__z_sz_ph_mapped = siam__z_sz_ph.tensor<double, 1>();
            siam__z_sz_ph.scalar<double>()() = z_sz;

            //cout << "D" << endl;

            Tensor filename(DT_STRING, TensorShape());
            //auto filename_mapped = filename.tensor<string, 1>();
            filename.scalar<string>()() = frame_name_list_[0];

            Tensor p3(DT_DOUBLE, {1});
            auto p3_m = p3.tensor<double, 1>();
            p3.scalar<double>()() = 3.0;

            //cout << "E" << endl;
            vector<tensorflow::Tensor> outputs;
            std::vector<pair<std::string, tensorflow::Tensor>> inputs = {{ "pos_x_ph", siam__pos_x_ph },
                                                                         { "pos_y_ph", siam__pos_y_ph },
                                                                         { "z_sz_ph", siam__z_sz_ph },
                                                                         { "filename", filename }};


            // stack_9 == template_z
            // ResizeBicubic == image
            status = session1->Run(inputs, {"mul","out__templates_z_"}, {}, &outputs);

            //session1->Close();

            //cout << "F" << endl;
            cout << outputs.size() << endl;
            cout << "image_       : " << outputs[0].DebugString() << endl;
            cout << "templates_z_ :" <<outputs[1].DebugString() << endl;
            //cout << frame_sz[0] << endl;

            // gelen ilk framelerin debug nanesi
            //float *ptr = outputs[0].flat<float>().data();
            // Mat cameraImg(frame_sz[1],frame_sz[0],CV_32FC3,ptr); // backup line


//            Mat cameraImg(frame_sz[1],frame_sz[0],CV_32FC3,ptr);
//            cameraImg = cameraImg / 255;
//            cvtColor(cameraImg,cameraImg,CV_BGR2RGB);
//            imshow("test",cameraImg);
//
//            waitKey(1);
            // end

            auto output_boxes = outputs[0].shaped<float, 4>({1, frame_sz[1],frame_sz[0], 3});

            //cout << output_boxes(0,300,250,0) << endl;
            //cout << output_boxes(0,300,250,1) << endl;
            //cout << output_boxes(0,300,250,2) << endl;

            if (!status.ok())
            {
                cout << status.ToString() << "\n";
                return 1;
            }

            Tensor image_ = outputs[0];
            Tensor templates_z_ = outputs[1];
            auto templates_z_mapped = templates_z_.tensor<float, 4>();

            // # 16
            cout << "###################### 16" << endl;
            //Tensor new_templates_z_ = templates_z_;
            Tensor new_templates_z_(DT_FLOAT,{3,17,17,32});
            auto new_templates_z_mapped = new_templates_z_.tensor<float, 4>();


            clock_t t_start;
            t_start = clock();
            cout << "t_start : " << t_start << endl;

            for (int j = 1; j < num_frames; ++j) {
                // # 17
                cout << "###################### 17" << endl;
                //cout << j << endl;
                vector<double> scaled_exemplar(scale_factors);
                transform(scaled_exemplar.begin(),scaled_exemplar.end(),scaled_exemplar.begin(),bind2nd(multiplies<double>(),z_sz));

                vector<double> scaled_search_area(scale_factors);
                transform(scaled_search_area.begin(),scaled_search_area.end(),scaled_search_area.begin(),bind2nd(multiplies<double>(),x_sz));

                vector<double> scaled_target_w(scale_factors);
                transform(scaled_target_w.begin(),scaled_target_w.end(),scaled_target_w.begin(),bind2nd(multiplies<double>(),target_w));

                vector<double> scaled_target_h(scale_factors);
                transform(scaled_target_h.begin(),scaled_target_h.end(),scaled_target_h.begin(),bind2nd(multiplies<double>(),target_h));

                //cout << scale_factors[0] << endl;
                //cout << scaled_exemplar[0] << endl;
                //cout << endl;
                // # 18
                cout << "###################### 18" << endl;

                Tensor scaled_search_area0(DT_DOUBLE,TensorShape());
                scaled_search_area0.scalar<double>()() = scaled_search_area[0];

                Tensor scaled_search_area1(DT_DOUBLE,TensorShape());
                scaled_search_area1.scalar<double>()() = scaled_search_area[1];

                Tensor scaled_search_area2(DT_DOUBLE,TensorShape());
                scaled_search_area2.scalar<double>()() = scaled_search_area[2];

                Tensor frame_name_list_j(DT_STRING,TensorShape());
                frame_name_list_j.scalar<string>()() = frame_name_list_[j];

                vector<tensorflow::Tensor> outputs2;
                std::vector<pair<std::string, tensorflow::Tensor>> inputs2 = {
                        { "pos_x_ph", siam__pos_x_ph },
                        { "pos_y_ph", siam__pos_y_ph },
                        { "x_sz0_ph", scaled_search_area0},
                        { "x_sz1_ph", scaled_search_area1 },
                        { "x_sz2_ph", scaled_search_area2 },
                        { "out__templates_z_", templates_z_}, //bu templates_z_ squeezed
                        // gonderilecek 3,17,17,32 oldugu icin simdilik gerek yok
                        { "filename",  frame_name_list_j}
                };


                // stack_9 == template_z
                // ResizeBicubic == image
                status = session1->Run(inputs2, {"mul","out__scores_"}, {}, &outputs2);

                if (!status.ok())
                {
                    cout << status.ToString() << "\n";
                    return 1;
                }

                cout << outputs2.size() << endl;
                cout << outputs2[0].DebugString() << endl;
                cout << outputs2[1].DebugString() << endl;

                Tensor image_ = outputs2[0];
                Tensor pre_scores_ = outputs2[1];
                auto pre_scores_mapped = pre_scores_.tensor<float, 4>();

                //cout << endl;
                //cout << "SHAPE_BEFORE" << endl;
                //cout << pre_scores_.DebugString() << endl;
                // # 19
                cout << "###################### 19" << endl;

                //cout << scores_.shape() << endl;
//                cout << pre_scores_.shape().dims() << endl;
//                cout << pre_scores_.shape().dim_size(0) << endl;
//                cout << pre_scores_.shape().dim_size(1) << endl;
//                cout << pre_scores_.shape().dim_size(2) << endl;
//                cout << pre_scores_.shape().dim_size(3) << endl;

                // TODO burasi squeeze olayini cozecek yer
                // FIXME squeeze fonksiyonu

                // squeezed olan
                Tensor scores_(DT_FLOAT,{3,257,257});
                auto scores_mapped = scores_.tensor<float, 3>();

                for (int gg = 0; gg < pre_scores_.shape().dim_size(0); ++gg) {
                    for (int i = 0; i < pre_scores_.shape().dim_size(1); ++i) {
                        for (int k = 0; k < pre_scores_.shape().dim_size(2); ++k) {
                            scores_mapped(gg,i,k) = pre_scores_mapped(gg,i,k,0);
                        }
                    }
                }

                //cout << "SHAPE_AFTER" << endl;
                //cout << scores_.DebugString() << endl;

                // penalize change of scale
                for (int l = 0; l < 257; ++l) {
                    for (int i = 0; i < 257; ++i) {
                        scores_mapped(0,l,i) = scores_mapped(0,l,i) * __hp__scale_penalty;
                        scores_mapped(2,l,i) = scores_mapped(2,l,i) * __hp__scale_penalty;
                    }
                }

                //cout << "SHAPE_AFTER_PARTY" << endl;
                //cout << scores_.DebugString() << endl;

                // TODO burdaki sayilarin dogrulugu ilerde kontrol edilmesi gerekir hale donusebilir

                double max[3];
                max[0] = -99999999;
                max[1] = -99999999;
                max[2] = -99999999;

                // FIXME needs optimization and axes precision
                // find scale with highest peak (after penalty)

                // in numpy np.amax(x, axis(1,2) means 3,257,257 lik matrisin
                // 0,257,257 1,257,257 2,257,257 seklinde birer max bulunmasini kapsiyor
                // optimize edilmemis hali ile uc for asagida cozuyor

                // for depth 0
                for (int i1 = 0; i1 < 257; ++i1) {
                    for (int i = 0; i < 257; ++i) {
                        if(scores_mapped(0,i1,i) > max[0]){
                            max[0] = scores_mapped(0,i1,i);
                        }
                    }
                }
                // for depth 1
                for (int i1 = 0; i1 < 257; ++i1) {
                    for (int i = 0; i < 257; ++i) {
                        if(scores_mapped(1,i1,i) > max[1]){
                            max[1] = scores_mapped(1,i1,i);
                        }
                    }
                }
                // for depth 2
                for (int i1 = 0; i1 < 257; ++i1) {
                    for (int i = 0; i < 257; ++i) {
                        if(scores_mapped(2,i1,i) > max[2]){
                            max[2] = scores_mapped(2,i1,i);
                        }
                    }
                }

                //cout << endl;
                //cout << max[0] << endl;
                //cout << max[1] << endl;
                //cout << max[2] << endl;
                //cout << endl;

                // find the index of the maximum element
                int new_scale_id = distance(max,max_element(max,max + 3));

                //cout << "DEBUG :: new_scale_id = " << new_scale_id << endl;

                // # 20
                cout << "###################### 20" << endl;
                // update scaled sizes
                x_sz = (1-__hp__scale_lr) * x_sz + __hp__scale_lr * scaled_search_area[new_scale_id];
                target_w = (1-__hp__scale_lr) * target_w + __hp__scale_lr * scaled_target_w[new_scale_id];
                target_h = (1-__hp__scale_lr) * target_h + __hp__scale_lr * scaled_target_h[new_scale_id];

                //cout << "DEBUG :: x_sz = " << x_sz << endl;
                //cout << "DEBUG :: target_w = " << target_w << endl;
                //cout << "DEBUG :: x_sz = " << target_h << endl;


                // # 21
                cout << "###################### 21" << endl;
                // select response with_scale_id
                Tensor score_(DT_FLOAT,{1,257,257});
                auto score_mapped = score_.tensor<float, 3>();

                // python : score_ = scores_[new_scale_id,:,:]
                for (int m = 0; m < 257; ++m) {
                    for (int i = 0; i < 257; ++i) {
                        score_mapped(0,m,i) = scores_mapped(new_scale_id,m,i);
                    }
                }

                //cout << "SHAPE of Score step 1 : " << endl;
                //cout << score_.DebugString() << endl;

                // python np.min(score_)
                double min = 9999999999;

                // for depth 0
                for (int i1 = 0; i1 < 257; ++i1) {
                    for (int i = 0; i < 257; ++i) {
                        if(score_mapped(0,i1,i) < min){
                            min = score_mapped(0,i1,i);
                        }
                    }
                }

                //cout << "min : " << min << endl;

                for (int i1 = 0; i1 < 257; ++i1) {
                    for (int i = 0; i < 257; ++i) {
                        score_mapped(0,i1,i) = score_mapped(0,i1,i) - min;
                    }
                }

                //cout << "score_ - np.min(score_) : " << score_.DebugString() << endl;

                double sum;

                for (int i1 = 0; i1 < 257; ++i1) {
                    for (int i = 0; i < 257; ++i) {
                        sum += score_mapped(0,i1,i);
                    }
                }

                //cout << "sum : " << sum << endl;

                for (int i1 = 0; i1 < 257; ++i1) {
                    for (int i = 0; i < 257; ++i) {
                        score_mapped(0,i1,i) = score_mapped(0,i1,i) / sum;
                    }
                }

                //cout << "score_ / np.sum(score_) : " << score_.DebugString() << endl;

                // # apply displacement penalty

                Tensor penalty_tensor(DT_DOUBLE,{257,257});
                auto penalty_tensor_mapped = penalty_tensor.tensor<double, 2>();

                for (int i1 = 0; i1 < 257; ++i1) {
                    for (int i = 0; i < 257; ++i) {
                        penalty_tensor_mapped(i1,i) = penalty(i1,i);
                    }
                }

                //cout << "penalty debug    0 : " << penalty_tensor.DebugString() << endl;
                //cout << "penalty debug  1,1 : " << penalty_tensor_mapped(1,1) << endl;
                //cout << "penalty debug 13,2 : " << penalty_tensor_mapped(2,2) << endl;
                //cout << "penalty debug 2,13 : " << penalty_tensor_mapped(3,3) << endl;

                double sum_penalty;

                for (int i1 = 0; i1 < 257; ++i1) {
                    for (int i = 0; i < 257; ++i) {
                        sum_penalty += penalty_tensor_mapped(i1,i);
                    }
                }

                //cout << "sum_penalty : " << sum_penalty << endl;

                for (int i1 = 0; i1 < 257; ++i1) {
                    for (int i = 0; i < 257; ++i) {
                        penalty_tensor_mapped(i1,i) = penalty_tensor_mapped(i1,i) / sum_penalty;
                    }
                }
                //cout << endl;
                //cout << endl;
                //cout << endl;
                //cout << "penalty debug    0 : " << penalty_tensor.DebugString() << endl;
                //cout << "penalty debug  1,1 : " << penalty_tensor_mapped(1,1) << endl;
                //cout << "penalty debug 13,2 : " << penalty_tensor_mapped(2,2) << endl;
                //cout << "penalty debug 2,13 : " << penalty_tensor_mapped(3,3) << endl;


                // # apply displacement penalty devami

                for (int i1 = 0; i1 < 257; ++i1) {
                    for (int i = 0; i < 257; ++i) {
                        score_mapped(0,i1,i) = (1-__hp__window_influence) * score_mapped(0,i1,i) + __hp__window_influence * penalty_tensor_mapped(i1,i);
                    }
                }
                //cout << endl;
                //cout << endl;
                //cout << "score_ - np.min(score_) : " << score_.DebugString() << endl;


                // # 22
                cout << "###################### 22" << endl;

                cout << "Update target positions" << endl;
                cout << " => pos_x                  : " << pos_x << endl;
                cout << " => pos_y                  : " << pos_y << endl;
                cout << " => score_                 : " << score_.DebugString() << endl;
                cout << " => final_score_sz         : " << final_score_sz << endl;
                cout << " => design.tot_stride      : " << __design__tot_stride << endl;
                cout << " => design.search_sz       : " << __design__search_sz << endl;
                cout << " => hp.response_up         : " << __hp__response_up << endl;
                cout << " => x_sz                   : " << x_sz << endl;

                // pos_x, pos_y = _update_target_position(pos_x, pos_y, score_, final_score_sz, design.tot_stride, design.search_sz, hp.response_up, x_sz)

                // # 23
                cout << "###################### 23" << endl;
                // # find location of score maximizer
                double duzScore[66049];

                int index=0;
                for (int i1 = 0; i1 < 257; ++i1) {
                    for (int i = 0; i < 257; ++i) {
                        duzScore[index++] = score_mapped(0,i1,i);
                    }
                }

                int score_argmax = distance(duzScore,max_element(duzScore,duzScore + 66049));
                cout << "score_argmax : " << score_argmax << endl;

                // np.unravel index nanesi
                double cmp_score = -9999999999;
                int i_coor;
                int j_coor;
                for (int i1 = 0; i1 < 257; ++i1) {
                    for (int i = 0; i < 257; ++i) {
                        if(score_mapped(0,i1,i) > cmp_score){
                            i_coor = i1;
                            j_coor = i;
                            cmp_score = score_mapped(0,i1,i);
                        }
                    }
                }

                cout << " i coor : " << i_coor << endl;
                cout << " j coor : " << j_coor << endl;

                // # find location of score maximizer cont'd
                int p[2];
                p[0] = i_coor;
                p[1] = j_coor;

                // # displacement from the center in search area final representation ...
                float center = float(final_score_sz - 1) / 2;
                //cout << "center : " << center << endl;


                double disp_in_area[2];
                disp_in_area[0] = p[0] - center;
                disp_in_area[1] = p[1] - center;

                // displacement from the center in instance crop

                double disp_in_xcrop[2];
                disp_in_xcrop[0] = disp_in_area[0] * float(__design__tot_stride) / __hp__response_up;
                disp_in_xcrop[1] = disp_in_area[1] * float(__design__tot_stride) / __hp__response_up;

                // displacement from the center in instance crop (in frame coordinates)

                double disp_in_frame[2];
                disp_in_frame[0] = disp_in_xcrop[0] * x_sz / __design__search_sz;
                disp_in_frame[1] = disp_in_xcrop[1] * x_sz / __design__search_sz;

                // *position* within frame in frame coordinates

                pos_x = pos_x + disp_in_frame[1];
                pos_y = pos_y + disp_in_frame[0];

                cout << "pos_x updated : " << pos_x << endl;
                cout << "pos_y updated : " << pos_y << endl;


                // # 24
                cout << "###################### 24" << endl;

                bboxes(j,0) = pos_x - target_w / 2;
                bboxes(j,1) = pos_y - target_h / 2;
                bboxes(j,2) = target_w;
                bboxes(j,3) = target_h;

                cout << j << endl;
                cout << bboxes(j,0) << endl;
                cout << bboxes(j,1) << endl;
                cout << bboxes(j,2) << endl;
                cout << bboxes(j,3) << endl;


                // # 25
                cout << "###################### 25" << endl;

                Tensor z_sz_tensor(DT_DOUBLE,TensorShape());
                z_sz_tensor.scalar<double>()() = z_sz;

                //cout << "pos_x  : " << pos_x << endl;
                //cout << "siam_pos_x  : " << siam__pos_x_ph.DebugString() << endl;
                //cout << "pos_y  : " << pos_y << endl;
                //cout << "z_sz   : " << z_sz << endl;
                //cout << "image_ : " << image_.DebugString() << endl;

                siam__pos_x_ph.scalar<double>()() = pos_x;
                siam__pos_y_ph.scalar<double>()() = pos_y;


                vector<tensorflow::Tensor> outputs3;
                std::vector<pair<std::string, tensorflow::Tensor>> inputs3 = {
                        { "pos_x_ph", siam__pos_x_ph},
                        { "pos_y_ph", siam__pos_y_ph },
                        { "z_sz_ph", z_sz_tensor},
                        { "mul",  image_}
                };

                // templates_z_ ve new_templates_z_ map isleri





                // # update the target representation with a rolling average
                if (__hp__z_lr > 0){

                    status = session1->Run(inputs3, {"out__templates_z_"}, {}, &outputs3);

                    if (!status.ok())
                    {
                        cout << status.ToString() << "\n";
                        return 1;
                    }

                    Tensor new_templates_z_temp(outputs3[0]);
                    auto new_templates_z_temp_mapped = new_templates_z_temp.tensor<float,4>();

                    //cout << new_templates_z_temp.DebugString() << endl;
                    //cout << new_templates_z_temp_mapped(0,0,0,0) << endl;


                    //cout << outputs3.size() << endl;
                    //cout << outputs3[0].DebugString() << endl;
                    //cout << new_templates_z_.DebugString() << endl;
                    //new_templates_z_ = outputs3[0];
                    //cout << new_templates_z_.DebugString() << endl;
                    //cout << new_templates_z_mapped(0,0,0,0) << endl;

                    for (int i1 = 0; i1 < 3; ++i1) {
                        for (int i2 = 0; i2 < 17; ++i2) {
                            for (int i3 = 0; i3 < 17; ++i3) {
                                for (int i4 = 0; i4 < 32; ++i4) {
//                                    cout << "= = = = > > > > DEBUG" << endl;
//                                    cout << endl;
//                                    cout << "i1 : " << i1 << endl;
//                                    cout << "i2 : " << i2 << endl;
//                                    cout << "i3 : " << i3 << endl;
//                                    cout << "i4 : " << i4 << endl;
//                                    cout << templates_z_mapped(i1,i2,i3,i4) << endl;
//                                    cout << new_templates_z_mapped(i1,i2,i3,i4) << endl;

                                    templates_z_mapped(i1,i2,i3,i4) = (1 - __hp__z_lr) * templates_z_mapped(i1,i2,i3,i4);
                                    new_templates_z_mapped(i1,i2,i3,i4) = __hp__z_lr * new_templates_z_temp_mapped(i1,i2,i3,i4);

//                                    cout << endl;
//                                    cout << templates_z_mapped(i1,i2,i3,i4) << endl;
//                                    cout << new_templates_z_mapped(i1,i2,i3,i4) << endl;
//                                    cout << "END CHUNK!" << endl;
                                }
                            }
                        }
                    }

                    for (int i1 = 0; i1 < 3; ++i1) {
                        for (int i2 = 0; i2 < 17; ++i2) {
                            for (int i3 = 0; i3 < 17; ++i3) {
                                for (int i4 = 0; i4 < 32; ++i4) {
                                    templates_z_mapped(i1,i2,i3,i4) = templates_z_mapped(i1,i2,i3,i4) + new_templates_z_mapped(i1,i2,i3,i4);
                                }
                            }
                        }
                    }

                    //cout << templates_z_.DebugString() << endl;
                    //cout << templates_z_mapped(0,0,0,1) << endl;

                    //cout << endl;

                }

                // # 26
                cout << "###################### 26" << endl;
                // update template patch size
                z_sz = (1 -__hp__scale_lr)*z_sz + __hp__scale_lr * scaled_exemplar[new_scale_id];
                cout << "z_sz updated : " << z_sz << endl;

                // # 27
                cout << "###################### 27" << endl;

//                float *ptr = outputs[0].flat<float>().data();
//                Mat cameraImg(frame_sz[1],frame_sz[0],CV_32FC3,ptr);
                float *ptr = image_.flat<float>().data();
                Mat cameraImg(frame_sz[1],frame_sz[0],CV_32FC3,ptr);
                if (__run__visualization){
//                    cout << "Come 10" << endl;
//                    cameraImg = cameraImg / 255;
//                    cvtColor(cameraImg,cameraImg,CV_BGR2RGB);
//                    imshow("test",cameraImg);
//                    waitKey(1);
                    cout << "Visualization" << endl;
                    cout << endl;

                    cameraImg = cameraImg / 255;
                    cvtColor(cameraImg,cameraImg,CV_BGR2RGB);

                    waitKey(1);
                    cout << "bboxes[j] : " << bboxes(j,0) << endl;
                    cout << "bboxes[j] : " << bboxes(j,1) << endl;
                    cout << "bboxes[j] : " << bboxes(j,2) << endl;
                    cout << "bboxes[j] : " << bboxes(j,3) << endl;
                    float vis_x = bboxes(j,0);
                    float vis_y = bboxes(j,1);
                    float vis_w = bboxes(j,0) + bboxes(j,2);
                    float vis_h = bboxes(j,1) + bboxes(j,3);

                    //Rect box = getBox(vis_x, xmin, ymax, xmax, frame_sz[1], frame_sz[0]);

//                    cout << box.tl() << endl;
//                    cout << box.br() << endl;
//                    rectangle(cameraImg, box.tl(), box.br(), CV_RGB(0, 0, 255), 4);
                    Point pt1(vis_x,vis_y);
                    Point pt2(vis_w,vis_h);
                    rectangle(cameraImg, pt1, pt2, Scalar(0,255,0), 2);
                    imshow("test",cameraImg);

                    cout << "frame_end : " << j << endl;
                    cout << endl;
                }
                //session1->Close();
                cout << endl;

            }
            //break;
        }
        //break;
    }
    return 0;
}
