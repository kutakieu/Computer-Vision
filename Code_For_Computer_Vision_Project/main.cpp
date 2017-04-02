#include "main.h"

#define IMAGE_HEIGHT 150
#define IMAGE_WIDTH  400
#define OUTPUT_IMAGE_HEIGHT 48.0
#define OUTPUT_IMAGE_WIDTH 48.0
#define DEBUG        false

// Random number genrator 
RNG rng(0xFFFFFFFF);



string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

vector<vector<int> > find_unique_contour_groups(Mat img, vector<vector<Point> > contours, vector<Vec4i> hierarchy)
{
    vector<vector<int> > groups;
    
    for(int i = 0; i < contours.size(); i++ )
    {
        vector<int> group;
        double totalArea = 0;
        Rect iBox = boundingRect(contours[i]);

        for (int j = 0; j < contours.size(); j++)
        {
            Rect jBox = boundingRect(contours[j]);
            double heightThreshold = (iBox.height + jBox.height) * 0.1;
            double widthRatio = 0;
            if (iBox.width > jBox.width)
                widthRatio = iBox.width / (double)jBox.width;
            else
                widthRatio = jBox.width / (double)iBox.width;
            if (abs(iBox.y - jBox.y) < heightThreshold && 
                abs(iBox.height - jBox.height) < heightThreshold &&
                widthRatio <= 2.8
                )
            {   
                group.push_back(j);
                totalArea += jBox.width * jBox.height;
            }
        }

        if(totalArea / group.size() < img.rows * img.cols / 2.0 && group.size() != 1)
        {
            bool alreadyHave = false;
            // int 
            for(int z = 0; z < groups.size(); z++)
            {
                int alreadyContain = -1;
                for(int m = 0 ; m < group.size(); m++)
                {
                    if(find(groups[z].begin(), groups[z].end(), group[m]) != groups[z].end()) 
                    {
                        alreadyContain = z;
                        // break;
                    }
                }
                if(alreadyContain > -1)
                {
                    for(int n = 0 ; n < group.size(); n++)
                    {
                        if(find(groups[z].begin(), groups[z].end(), group[n]) == groups[z].end()) 
                        {
                            groups[z].push_back(group[n]);
                        }
                    }
                    alreadyHave = true;
                    break;
                }
                if(group == groups[z])
                {
                    alreadyHave = true;
                    // break;
                }
                
            }

            if(!alreadyHave)
            {
                groups.push_back(group);
            }
        }
    }
    
    if (false)
    {
        for(int i = 0 ; i < groups.size(); i++)
        {

            Mat drawing = Mat::zeros(img.size(), CV_8UC3);
            double totoalArea = 0;
            cout << "Components: ";
            for(int j = 0 ; j < groups[i].size(); j++)
            {
                Rect box = boundingRect(contours[groups[i][j]]);
                totoalArea += box.height * box.width;
                cout << groups[i][j] << ", ";
                Scalar color = Scalar(rng.uniform(0,255), rng.uniform(0,255), rng.uniform(0,255));
                drawContours(drawing, contours, groups[i][j], color, 2, 8, hierarchy, 0, Point());
            }
            cout << endl;
            cout << "area: " << totoalArea/groups[i].size() << endl;
            imshow("drawing", drawing);
            waitKey(0);
        }
    }
    return groups;
}

vector<int> find_max_area_unique_contour_group(Mat img, vector<vector<Point> > contours, vector<Vec4i> hierarchy, vector<vector<int> > groups)
{
    double maxArea = -1; 
    vector<int> maxAreaGroup;
    for(int i = 0 ; i < groups.size(); i++)
    {
        double totoalArea = 0;
        for(int j = 0 ; j < groups[i].size(); j++)
        {
            Rect box = boundingRect(contours[groups[i][j]]);
            totoalArea += box.height * box.width;
        }
        if(totoalArea > maxArea)
        {
            maxAreaGroup = groups[i];
            maxArea = totoalArea;
        }
    }
    if (DEBUG)
    {
        Mat drawing = Mat::zeros( img.size(), CV_8UC3 );
        for(int i = 0 ; i < maxAreaGroup.size() ; i++)
        {
            Scalar color = Scalar(rng.uniform(0,255), rng.uniform(0,255), rng.uniform(0,255));
            drawContours(drawing, contours, maxAreaGroup[i], color, 2, 8, hierarchy, 0, Point());
        }
        imshow("find_max_area_unique_contour_group", drawing);
    }
    return maxAreaGroup;
}


vector<int> remove_child_contour(vector<vector<Point> > contours, vector<Vec4i> hierarchy, vector<int> group)
{       
    // [Next, Previous, First_Child, Parent]
    vector<int> newGroup;
    for(int i = 0 ; i < group.size(); i++)
    {
        bool isContain = false;
        Vec4i h = hierarchy[group[i]];
        int parent = h[3];
        for(int j = 0 ; j < group.size() ; j ++)
        {
            if (parent == group[j])
                isContain = true;
        }
        if(!isContain)
            newGroup.push_back(group[i]);
    }
    return newGroup;
}

Rect find_group_bounding_box(Mat img, vector<vector<Point> > contours, vector<Vec4i> hierarchy, vector<int> group)
{

    int minY = IMAGE_HEIGHT + 1;
    int maxY = -1;
    int minX = IMAGE_WIDTH + 1;
    int maxX = -1;
    vector<Point> pointsInGroup;
    for(int i = 0 ; i < group.size() ; i++)
    {
        Rect box = boundingRect(contours[group[i]]);
        if(box.y < minY)
            minY = box.y;
        if(box.y+box.height > maxY)
            maxY = box.y+box.height;
        if(box.x < minX)
            minX = box.x;
        if(box.x+box.width > maxX)
            maxX = box.x+box.width;
    }

    Rect rect = Rect(0, minY, IMAGE_WIDTH, maxY-minY);

    vector<int> removeContourList;

    for(int i = 0 ; i < contours.size() ; i++)
    {
        for(int j = 0 ; j < contours[i].size() ; j++)
        {
            int rowCnt = contours[i][j].y;
            int colCnt = contours[i][j].x;
            if(rowCnt < rect.y)
            {
                removeContourList.push_back(i);
                break;
            }
            if(rowCnt > rect.y + rect.height)
            {
                removeContourList.push_back(i);
                break;
            }
        }
    }

    int minArea = IMAGE_HEIGHT * IMAGE_WIDTH;
    int minAreaIndex = -1;
    // int maxX = IMAGE_WIDTH + 1;
    for(int i = 0 ; i < removeContourList.size() ; i ++)
    {
        Rect box = boundingRect(contours[removeContourList[i]]);
        int area = box.height * box.width;
        if(box.width * box.height > IMAGE_WIDTH * IMAGE_HEIGHT * 0.5)
        {
            if(area < minArea)
            {
                minArea = area;
                minAreaIndex = removeContourList[i];
            }
        }
    }
    if(minAreaIndex > -1)
    {
        Rect box = boundingRect(contours[minAreaIndex]);
        rect.x = box.x;
        rect.width = box.width;
    }



    if (DEBUG)
    {
        Mat drawing = img.clone();
        vector<Point> points;
        for(int i = 0 ; i < group.size() ; i++)
            for(int j = 0 ; j < contours[group[i]].size() ; j++)
                points.push_back(contours[group[i]][j]);
        RotatedRect rotatedBox = minAreaRect(cv::Mat(points));
        cv::Point2f vertices[4];
        rotatedBox.points(vertices);
        for(int i = 0; i < 4; ++i)
            cv::line(drawing, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 0, 0), 1, CV_AA);
        imshow("rotatedbox", drawing);
        imshow("find_group_bounding_box", img(rect));
    }

    return rect;
}


Mat image_deskew(Mat img, vector<vector<Point> > contours, vector<Vec4i> hierarchy, vector<int> group)
{   
    Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(img.clone(), img, MORPH_OPEN, element);
    vector<Point> points;
    for(int i = 0 ; i < group.size() ; i++)
        for(int j = 0 ; j < contours[group[i]].size() ; j++)
            points.push_back(contours[group[i]][j]);
    RotatedRect rotatedBox = minAreaRect(cv::Mat(points));
    double angle = rotatedBox.angle;
    if (angle < -45.)
        angle += 90.;
    points.clear();
    Mat_<uchar>::iterator it = img.begin<uchar>();
    Mat_<uchar>::iterator end = img.end<uchar>();
    for (; it != end; ++it)
        if (*it)
          points.push_back(it.pos());
    cv::RotatedRect box = cv::minAreaRect(cv::Mat(points));

    Mat rotMat = getRotationMatrix2D(box.center, angle, 1);
    Mat rotated, rotatedPadded;
    warpAffine(img, rotated, rotMat, img.size(), cv::INTER_CUBIC, cv::BORDER_CONSTANT, 255);
    copyMakeBorder(rotated, rotatedPadded, 10, 10, 0, 0, BORDER_CONSTANT, 255);
    if (DEBUG)
    {   
        imshow("rotatedPadded", rotatedPadded);
    }
    return rotatedPadded;
}

Mat revert_background(Mat img)
{
    int numBlack = countNonZero(img);
    int numWhite = img.cols * img.rows - numBlack;
    if (numBlack < numWhite)
         bitwise_not(img, img);
    
    return img;
}



Mat remove_border(Mat img, int radius)
{

    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;

    findContours(img.clone(), contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

    int imgRows = img.rows;
    int imgCols = img.cols; 
    
    vector<int>  removeList;

    int maxArea = -1;
    vector<Point> biggestContour;
    int index = 0;
    for(int i = 0 ; i < contours.size() ; i++)
    {   
        Rect box = boundingRect(contours[i]);
        int area = box.width * box.height;
        if(area > maxArea)
        {
            maxArea = area;
            biggestContour = contours[i];
            index = i;
            
        }

        for(int j = 0 ; j < contours[i].size() ; j++)
        {
            int rowCnt = contours[i][j].y;
            int colCnt = contours[i][j].x;
            bool check = ((colCnt >= 0) && (colCnt < radius)) || ((colCnt >= imgCols-1-radius) && (colCnt < imgCols));
            if (check)
            {
                removeList.push_back(i);
                break;
            }

        }
    }

    Mat mask = Mat::zeros(img.size(), CV_8UC1 );
    drawContours(mask, contours, index, 255, -1, 8, hierarchy, 0, Point());

    for (int i = 0 ; i < removeList.size(); i++)
    {
        if(removeList[i] != index)
        {
            drawContours(mask, contours, removeList[i], 0, -1, 8, hierarchy, 0, Point());
        }
    }

    bitwise_xor(img, mask, img);
    bitwise_not(img, img);


    if(DEBUG)
    {
        Mat drawing = Mat::zeros(img.size(), CV_8UC3);
        for(int i = 0 ; i < contours.size() ; i++)
        {
            Scalar color = Scalar(rng.uniform(0,255), rng.uniform(0,255), rng.uniform(0,255));
            drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point());
        }
        imshow("contours cropped", drawing);
        imshow("mask", mask);
        imshow("after mask", img);
    }

    return img;
}




string getFileName(const string& s) 
{

    char sep = '/';
    size_t i = s.rfind(sep, s.length());
    if (i != string::npos) {
        return (s.substr(i+1, s.length() - i));
        // size_t position = swx.find(".");
        // return (string::npos == position)? swx : swx.substr(0, position);
    }

    return("");
}

void process(string filename)
{
    Mat input;
    
    // Read in the image
    input = imread(filename, IMREAD_GRAYSCALE);
    
    // cout << "here" << endl;
    // intialize
    Mat resized; 
    // resize image
    resize(input, resized, Size(IMAGE_WIDTH, IMAGE_HEIGHT), 0, 0, INTER_AREA);
    // imshow("resized", resized);
    // waitKey(0);
    // median blur and gaussian blur
    medianBlur(resized, resized, 3);
    GaussianBlur(resized, resized, Size(3,3), 0.5);
    

    // apply Wolf binarization
    Mat binary(resized.rows, resized.cols, CV_8U);
    NiblackSauvolaWolfJolion(resized, binary, WOLFJOLION, 50, 100, 0.4, 128);
    
    // remove small bridges connecting border and charactors
    Mat element = getStructuringElement(MORPH_RECT, Size(7, 1));
    morphologyEx(binary.clone(), binary, MORPH_CLOSE, element);


    // imshow("binary", binary);

    // find contours
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;

    findContours(binary.clone(), contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

    // if (true)
    // {
    //     Mat drawing = Mat::zeros( binary.size(), CV_8UC3 );
    //     for(int i = 0 ; i < contours.size() ; i++)
    //     {
    //         Scalar color = Scalar(rng.uniform(0,255), rng.uniform(0,255), rng.uniform(0,255));
    //         drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point());
    //     }
    //     imshow("contours", drawing);
    // }
    // find unique contour groups based on its boundingbox's height, upper-left corner and width
    vector<vector<int> > groups = find_unique_contour_groups(binary, contours, hierarchy);

    // find the group with maximum avg area
    vector<int> maxAreaGroup = find_max_area_unique_contour_group(binary, contours, hierarchy, groups);

    // cropped the image
    Rect cropRect = find_group_bounding_box(binary, contours, hierarchy, maxAreaGroup);



    Mat cropped = binary(cropRect);

    // rever the backgrround
    cropped = revert_background(cropped);

    // deskew the image
    Mat rotated = image_deskew(cropped, contours, hierarchy, maxAreaGroup);

    // // remove border
    rotated = remove_border(rotated, 5);
    

    // find contours again
    contours.clear();
    hierarchy.clear();
    findContours(rotated.clone(), contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
    groups = find_unique_contour_groups(rotated, contours, hierarchy);
    maxAreaGroup = find_max_area_unique_contour_group(rotated, contours, hierarchy, groups);
    maxAreaGroup = remove_child_contour(contours, hierarchy, maxAreaGroup);

    std::sort(maxAreaGroup.begin(), maxAreaGroup.end(), Comparator(contours));

    for(int i = 0 ; i < maxAreaGroup.size() ; i++)
    {
        Rect box = boundingRect(contours[maxAreaGroup[i]]);
        double fy = (OUTPUT_IMAGE_HEIGHT - 10) > box.height ? box.height / (OUTPUT_IMAGE_HEIGHT - 10) : (OUTPUT_IMAGE_HEIGHT - 10) / box.height;
        // printf("fy %f\n", fy);
        Mat output = rotated(box);
        // printf("output %d output %d\n", output.cols, output.rows);
        resize(output, output, Size(0,0), fy, fy, INTER_AREA);
        // imshow("output", output);
        int padwidth = (OUTPUT_IMAGE_WIDTH - output.cols) / 2;
        int padheight = (OUTPUT_IMAGE_HEIGHT - output.rows) / 2;
        copyMakeBorder(output, output, padheight, padheight, padwidth, padwidth, BORDER_CONSTANT, 255);
        resize(output, output, Size(48,48), 0, 0, INTER_AREA);
        // imshow("im", output);
        // waitKey(0);
        stringstream out;
        out << i;
        string s = out.str();
        string outputpath = "output/" + getFileName(filename) + "/" + s + ".png";
        // cout << outputpath << endl;
        imwrite(outputpath, output);
    }

}

int main(int argc, char *argv[])
{	
    // cout << argc << endl;

    if(argc != 2)
        exit(0);

    cout << argv[1] << endl;
    process(argv[1]);


    // vector<string> v;

    // v.push_back("VIC.png");
    // v.push_back("WA.jpg");
    // v.push_back("SA.png");
    // v.push_back("california.jpg");
    // v.push_back("ACT.jpg");
    // v.push_back("WA1.jpg");
    // v.push_back("r1.jpg");
    // v.push_back("r2.jpg");
    // v.push_back("r7.jpg");
    // v.push_back("r8.jpg");
    // v.push_back("r9.jpg");
    // v.push_back("r6.jpg");
    // v.push_back("r5.png");
    // v.push_back("r3.jpg");
    // v.push_back("VIC2.jpg");
    // v.push_back("r4.jpg");
    // v.push_back("r10.jpg");
    // v.push_back("r13.jpg");


    // for(int i = 0 ; i < v.size() ; i++)
    // {
    //     process("images/"+v[i]);
    //     // waitKey(0);
    // }
    
    // waitKey(0);// wait for a keystroke in the window
    
    return 0;

}
