/*Amin Dorostanian - S005824 
Murat Ozcelik - S003125 
Ozyegin University, Department of Computer Science*/
#include "helper.h"
using namespace std;
using namespace cv;
struct Category
{
	int count;
	string name;	
	int ind;
} ;
struct Confusion
{
	int ind;
	string name;
	std::vector<Category> categories;
} ;
int main(int argc,char **argv){
	cout << "OpenCV Version = " << CV_MAJOR_VERSION << "." << CV_MINOR_VERSION << endl;
	std::vector<Mat>  train_set,test_set;
	std::vector<string> train_labels,test_labels;
	// Reading images
	string dataset_address;
	if (argc != 2) {
        // Tell the user how to run the program
		cerr << "Usage: You need to add dataset folder containing train folder and test folder as the input argument!" << endl;
		return 1;
	}else 
	dataset_address = argv[1];
	cout <<"Reading Training set\n";	
	readImages(dataset_address+"/train",train_set,train_labels);
	cout <<"Reading Testing set\n";	
	readImages(dataset_address+"/test",test_set,test_labels);	

	// Method 1 : Using Tiny Images and KNN
    Mat current_tiny_train,current_tiny_test;
	// First we need to resize images to 16*16
    Mat current_img;
    cout <<"===========================\n";
    cout <<"Tiny Image with KNN :\n";

   	int misclassified_count =0;
    for (int i=0;i<test_set.size();i++){
    	 resize(test_set.at(i), current_tiny_test, Size(16,16));
    	 // Storing each image as a vector
    	 current_tiny_test = current_tiny_test.reshape(0,1);
    	 // Finding nearest neighbour
    	 int current_best_ind =-1;
    	 float current_best_distance = 1000000;
    	 for (int j=0;j<train_set.size();j++){
    	 	 resize(train_set.at(j), current_tiny_train, Size(16,16));
    	 	// Storing each image as a vector
    		 current_tiny_train = current_tiny_train.reshape(0,1);
    		 //normalize(current_tiny_train, current_tiny_train, 0, 1, CV_MINMAX);
    		 //normalize(current_tiny_test, current_tiny_test, 0, 1, CV_MINMAX);

    	 	float dist = norm(current_tiny_train -current_tiny_test);

    	 	//cout <<" distance = " << dist << ", best index = " << current_best_ind << endl; getchar();
    	 	if (dist < current_best_distance ){
    	 		current_best_distance = dist;
    	 		current_best_ind = j;
    	 	}
    	 }

    	 if ( !isequals(test_labels.at(i) , train_labels.at(current_best_ind) ) ){
    	 	misclassified_count++;
    	 }
    	 cout << "Accuracy = " <<  100 * (1 - (misclassified_count/(double)i) )<<"\r" << std::flush;
    }
    cout << endl;

    // Method 2 : SIFT 
    cout << "===========================\n";
    cout << "Extracting SIFT Features of train set ...\n";

   	// Check if we already have vocabulary written in file
    Mat vocabulary;
    if( !readVocabulary( "../vocabulary.dat", vocabulary) )
    {
    	SiftFeatureDetector detector;
    	SiftDescriptorExtractor extractor;

    	Mat all_descriptors;
    	Mat descriptors;

    	for (int i=0;i<train_set.size();i++){
    		vector<cv::KeyPoint> keypoints;
    		detector.detect(train_set.at(i), keypoints);
    		extractor.compute(train_set.at(i), keypoints, descriptors);
    		all_descriptors.push_back(descriptors);
    		cout << "Extracting image #"<<i << "/" << train_set.size() << "\r" << std::flush;
    	}
    	cout <<endl << all_descriptors.size() << " features extracted for train set.\n";                   
    // Now cluster SIFT features using KNN
    	int vocabulary_size = 70;
    // Clustering all SIFT descriptors
    	TermCriteria terminate_criterion;
    	terminate_criterion.epsilon = FLT_EPSILON;
    	BOWKMeansTrainer bowTrainer( vocabulary_size, terminate_criterion, 3, KMEANS_PP_CENTERS );
    	for (int i=0;i<all_descriptors.size().height;i++){
    		Mat current_descriptor = all_descriptors.row(i);
    	//cout << "Size of current_descriptor = " << current_descriptor.size() << endl;  getchar();
    		bowTrainer.add(current_descriptor);
    		cout << "Adding Feature #" << i << " to Bag-Of-Words K-Means Trainer ...  \r" << std::flush;
    	}
    	cout << "\nClustering... Please Wait ...\n";
    	vocabulary = bowTrainer.cluster();

    	cout << "\nSIFT Features Clustered in " << vocabulary.size() << " clusters." << endl;
    	if( !writeVocabulary("../vocabulary.dat", vocabulary) )
    	{
    		cout << "Error: file " << "../vocabulary.dat" << " can not be opened to write" << endl;
    		exit(-1);
    	}
    }else
    cout << "Visual Vocabulary read from file successfully!\n";
    // Building Histograms
    cout << "===========================\n";

    std::vector< DMatch > matches;
  	// Matching centroids with training set
    std::vector<DMatch> trainin_set_matches;

    Ptr<FeatureDetector> featureDetector = FeatureDetector::create( "SIFT" );
    Ptr<DescriptorExtractor> descExtractor = DescriptorExtractor::create( "SIFT" );
    Ptr<BOWImgDescriptorExtractor> bowExtractor;

    Ptr<DescriptorMatcher> descMatcher = DescriptorMatcher::create( "BruteForce" );

    bowExtractor = new BOWImgDescriptorExtractor( descExtractor, descMatcher );

    bowExtractor->setVocabulary(vocabulary);


    Mat train_hist,test_hist;
    cout << "Building Histograms for training set :\n";   
    if( !readVocabulary( "../train_hist.dat", train_hist) )
    {
    	for (int i=0;i<train_set.size();i++){
    		vector<cv::KeyPoint> keypoints;
  		// Each descriptor is histogram for the image
    		Mat descriptors;
    		featureDetector->detect( train_set.at(i), keypoints );
    		bowExtractor->compute( train_set.at(i), keypoints, descriptors);
    		train_hist.push_back(descriptors);
    		cout << "Image #" << i <<" Bag-Of-Words descriptors extracted , size = " << descriptors.size() << "\r" << std::flush;
    	}
    	writeVocabulary("../train_hist.dat", train_hist);
    }else{
    cout << "Train Histograms read from file successfully!\n";
    }

    cout << "\nBuilding Histograms for test set :\n";   
    if( !readVocabulary( "../test_hist.dat", test_hist) )
    {
    	for (int i=0;i<test_set.size();i++){
    		vector<cv::KeyPoint> keypoints;
  		// Each descriptor is histogram for the image
    		Mat descriptors;
    		featureDetector->detect( test_set.at(i), keypoints );
    		bowExtractor->compute( test_set.at(i), keypoints, descriptors);
    		test_hist.push_back(descriptors);
    		cout << "Image #" << i <<" Bag-Of-Words descriptors extracted , size = " << descriptors.size() << "\r" << std::flush;
    	}
    	writeVocabulary("../test_hist.dat", test_hist);
    }else{
    cout << "Test Histograms read from file successfully!\n";
    }


    cout << "\n===========================\n";
    cout << "Classifying using SIFT and K-Nearest Neighbour:\n";

     misclassified_count = 0;
    for (int i=0;i<test_set.size();i++){
    	Mat current_test_hist = test_hist.row(i);
    	int current_best_ind =-1;
    	float current_best_distance = 100000;
    	for (int j=0;j<train_set.size();j++){
    		Mat current_train_hist = train_hist.row(j);
    		float dist = norm(current_train_hist -current_test_hist);
    		if (dist < current_best_distance ){
    			current_best_distance = dist;
    			current_best_ind = j;
    		}
    	}

    	if ( !isequals(test_labels.at(i) , train_labels.at(current_best_ind) ) ){
    		misclassified_count++;
    	}
    	cout << "Accuracy = " << RED << 100 * (1 - misclassified_count/(double)i )<< Color_Off<< "\r" << std::flush;

    }
    cout << endl;

    cout << "\n===========================\n";
    cout << "Classifying using SVM :\n";

    // Set up SVM's parameters
   // (int svm_type, int kernel_type, double degree, double gamma, 
   // 	double coef0, double Cvalue, double nu, double p, CvMat* class_weights, CvTermCriteria term_crit)
    CvSVMParams params;
    params.svm_type    = CvSVM::C_SVC;
    params.kernel_type = CvSVM::LINEAR;
    params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 1000, 1e-7);
    params.C = 120;
    //params.nu = 0.5;

    cout << "SMV is training ... Please wait ...\n";
    // Train  15  one-vs-all SVMs
    std::vector<Ptr<CvSVM> > svms;
    CvSVM  *mainSVM;
    mainSVM= new CvSVM;
    for (int cat=0;cat<15;cat++){

    	// Construct labels_mat for each SVM
    	Mat labels_mat = Mat::zeros(train_set.size(),1,CV_32F);
    	for (int i=0;i<train_set.size();i++){
    		int current_label;
    		int lbl = i/100;
    		if ( lbl == cat)
    			current_label = -1;
    		else 
    			current_label = 1;
    		labels_mat.at<float>(i) = current_label;
    	}
   	    CvSVM *current_svm ;
   	    current_svm= new CvSVM;
    	current_svm->train(train_hist, labels_mat,Mat(),Mat(),params);
   		svms.push_back(current_svm);
    }
    Mat labels_mat_main = Mat::zeros(train_set.size(),1,CV_32F);
    	for (int i=0;i<train_set.size();i++){
    		int lbl = i/100;
    		labels_mat_main.at<float>(i) = lbl;
    	}
    mainSVM->train(train_hist,labels_mat_main,Mat(),Mat(),params);
    cout << "SVM training done!\n";


    // Building Confusion Matrix
	std::vector<Confusion> all_confs;
	for (int i=0;i<15;i++){
		Confusion currenct_conf ;
		currenct_conf.ind = i;
		currenct_conf.name = train_labels.at(i*100);

		std::vector<Category> cats;
		for (int j=0;j<15;j++){
			Category current_cat;
			current_cat.ind = j;
			current_cat.name =  train_labels.at(j*100);
			current_cat.count = 0;
			cats.push_back(current_cat);
		}
		currenct_conf.categories = cats;
		all_confs.push_back(currenct_conf);
	}
    // Test SVM
    misclassified_count = 0;
    for (int i=0;i<test_set.size();i++){

    	int best_cat = -1;
    	float best_response=-1000000000;

    	for (int ind=0;ind<15;ind++){
    		float response = (svms.at(ind)->predict(test_hist.row(i) , true));
    		if (response > best_response){
    			best_cat  = ind;
    			best_response = response;
    		}
    	}
    	 best_cat = mainSVM->predict(test_hist.row(i) ,false );
    	 if ( !isequals(test_labels.at(i) , train_labels.at(best_cat * 100) ) ){
    	 	misclassified_count++;
    	 }

    	 int test_img_ind =-1;
    	 for (int k=0;k<train_set.size();k=k+100){
    	 	if (isequals(test_labels.at(i) , train_labels.at(k) )){
    	 		test_img_ind = k/100;
    	 		break;
    	 	}
    	 }
    	 all_confs.at(test_img_ind).categories.at(best_cat).count++;

    	 cout << "Accuracy = " << RED << 100 * (1 - misclassified_count/(double)i )<< Color_Off << "\r" << std::flush;
    }
    cout << endl;
    cout << "Writing Results into files...\n";
    // Printing Results and Building HeatMap
    float data[15][15];
    float max_num=0;

    ofstream result_file("../res.csv");
    for (int i=0;i<all_confs.size() ; i++){
    	Confusion current_conf = all_confs.at(i);
    	result_file <<  current_conf.name ;


    	for (int j=0;j<current_conf.categories.size();j++){
    		Category current_cat = current_conf.categories.at(j);
    		result_file << "," << current_cat.count;
    		data[i][j] = current_cat.count;
    		if (data[i][j] > max_num)
    			max_num = data[i][j];
    	}
    	result_file << "\n";
    }
    result_file.close();

    // normalize data
  /*  for (int i=0;i<15;i++)
    	for (int j=0;j<15;j++)
    		data[i][j] = data[i][j]/max_num;
    


    Mat heatmap = Mat::zeros(15 , 15 ,CV_32FC3);
    for (int i=0;i<15;i++)
    	for (int j=0;j<15;j++){
    		heatmap.at<Vec3b>(j,i)[0] = data[i][j];
    		heatmap.at<Vec3b>(j,i)[1] = 1;
    		heatmap.at<Vec3b>(j,i)[2] = 1;
    	}

    	imwrite("../heat.jpg",heatmap);*/
    return 0;
}