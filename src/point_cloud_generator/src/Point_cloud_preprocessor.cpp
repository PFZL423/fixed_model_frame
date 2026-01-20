#include "point_cloud_generator/Point_cloud_preprocessor.h"
#include <iostream>

PointCloudPreprocessor::PointCloudPreprocessor(const PreprocessingParams &params): params_(params)
{
    std::cout << "[DEBUG Preprocessor] Constructor start" << std::endl;
    std::cout << "[DEBUG Preprocessor] Setting voxel filter leaf size: " 
              << params_.leaf_size_x << ", " << params_.leaf_size_y << ", " << params_.leaf_size_z << std::endl;
    
    voxel_filter_.setLeafSize(params_.leaf_size_x, params_.leaf_size_y, params_.leaf_size_z);
    
    if(params_.use_outlier_removal)
    {
        std::cout << "[DEBUG Preprocessor] Setting SOR filter: mean_k=" << params_.sor_mean_k 
                  << ", stddev_mul_thresh=" << params_.sor_stddev_mul_thresh << std::endl;
        sor_filter_.setMeanK(params_.sor_mean_k);
        sor_filter_.setStddevMulThresh(params_.sor_stddev_mul_thresh);
    }
    std::cout << "[DEBUG Preprocessor] Constructor complete" << std::endl;
}

bool PointCloudPreprocessor::process(const PointCloudConstPtr &raw_cloud,
                                     PointCloudPtr &processed_cloud_out)
{
    std::cout << "[DEBUG Preprocessor] Process start" << std::endl;
    
    if (!raw_cloud || raw_cloud->empty())
    {
        std::cerr << "Error: Input raw cloud is null or empty." << std::endl;
        return false;
    }
    
    std::cout << "[DEBUG Preprocessor] Input cloud size: " << raw_cloud->size() << std::endl;
    std::cout << "[DEBUG Preprocessor] Input cloud ptr address: " << raw_cloud.get() << std::endl;
    std::cout << "[DEBUG Preprocessor] Creating downsampled_cloud" << std::endl;
    
    // 使用安全的方法创建点云，避免 boost::shared_ptr 拷贝构造问题
    PointCloudPtr downsampled_cloud(new PointCloud());
    std::cout << "[DEBUG Preprocessor] downsampled_cloud created at: " << downsampled_cloud.get() << std::endl;
    
    std::cout << "[DEBUG Preprocessor] Setting input cloud for voxel filter" << std::endl;
    voxel_filter_.setInputCloud(raw_cloud);
    
    std::cout << "[DEBUG Preprocessor] Applying voxel filter" << std::endl;
    voxel_filter_.filter(*downsampled_cloud);
    
    std::cout << "[DEBUG Preprocessor] Voxel filter complete, result size: " << downsampled_cloud->size() << std::endl;
    
    if (downsampled_cloud->empty())
    {
        std::cerr << "Warning: Cloud is empty after downsampling." << std::endl;
        // 即使为空，也应该清理输出，并返回true表示流程正常结束
        processed_cloud_out->clear();
        return true;
    }
    
    if(params_.use_outlier_removal)
    {
        std::cout << "[DEBUG Preprocessor] Applying outlier removal" << std::endl;
        PointCloudPtr filtered_cloud(new PointCloud());
        std::cout << "[DEBUG Preprocessor] filtered_cloud created at: " << filtered_cloud.get() << std::endl;
        
        std::cout << "[DEBUG Preprocessor] Setting input cloud for SOR filter" << std::endl;
        sor_filter_.setInputCloud(downsampled_cloud);
        std::cout << "[DEBUG Preprocessor] SOR filter input set" << std::endl;
        
        std::cout << "[DEBUG Preprocessor] Applying SOR filter" << std::endl;
        sor_filter_.filter(*filtered_cloud);
        std::cout << "[DEBUG Preprocessor] SOR filter complete, result size: " << filtered_cloud->size() << std::endl;
        
        if (filtered_cloud->empty())
        {
            std::cerr << "Warning: Cloud is empty after outlier removal." << std::endl;
            processed_cloud_out->clear();
            return true;
        }
        
        std::cout << "[DEBUG Preprocessor] Final copy (filtered) to output" << std::endl;
        std::cout << "[DEBUG Preprocessor] Output cloud ptr address: " << processed_cloud_out.get() << std::endl;
        
        // 直接使用安全的 copyPointCloud 将结果拷贝到输出
        pcl::copyPointCloud(*filtered_cloud, *processed_cloud_out);
    }
    else
    {
        std::cout << "[DEBUG Preprocessor] Final copy (downsampled) to output" << std::endl;
        std::cout << "[DEBUG Preprocessor] Output cloud ptr address: " << processed_cloud_out.get() << std::endl;
        
        // 直接使用安全的 copyPointCloud 将下采样结果拷贝到输出
        pcl::copyPointCloud(*downsampled_cloud, *processed_cloud_out);
    }
    
    std::cout << "[DEBUG Preprocessor] Final copy complete, output size: " << processed_cloud_out->size() << std::endl;
    
    std::cout << "Preprocessing finished. Raw: " << raw_cloud->size()
              << ", Processed: " << processed_cloud_out->size() << std::endl;

    std::cout << "[DEBUG Preprocessor] Process complete, returning true" << std::endl;
    return true;
}