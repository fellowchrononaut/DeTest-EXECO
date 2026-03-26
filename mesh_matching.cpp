// mesh_matching.cpp — robust matching + your original visualization style
// Build deps (as you already have): pcl_common pcl_io pcl_features pcl_keypoints
// pcl_registration pcl_visualization pcl_filters yaml-cpp lz4

#include <iostream>
#include <string>
#include <vector>
#include <cmath>

#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>

#include <pcl/io/ply_io.h>
#include <pcl/io/obj_io.h>
#include <pcl/PolygonMesh.h>
#include <pcl/conversions.h>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/search/kdtree.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <pcl/filters/voxel_grid.h>

#include <pcl/features/normal_3d_omp.h>
#include <pcl/keypoints/iss_3d.h>
#include <pcl/keypoints/harris_3d.h>

#include <pcl/features/fpfh_omp.h>
#include <pcl/features/shot_omp.h>

#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/correspondence_rejection_one_to_one.h>

#include <pcl/common/common_headers.h>
#include <pcl/common/transforms.h>
#include <pcl/common/point_tests.h> // pcl::isFinite
#include <pcl/visualization/pcl_visualizer.h>

// --- types ---
using PointT           = pcl::PointXYZ;
using PointCloudT      = pcl::PointCloud<PointT>;
using NormalT          = pcl::Normal;
using NormalCloudT     = pcl::PointCloud<NormalT>;
using FPFHDescriptorT  = pcl::FPFHSignature33;
using FPFHCloudT       = pcl::PointCloud<FPFHDescriptorT>;
using SHOTDescriptorT  = pcl::SHOT352;
using SHOTCloudT       = pcl::PointCloud<SHOTDescriptorT>;

template <typename DescT>
struct DescAndKp {
  typename pcl::PointCloud<DescT>::Ptr desc; // filtered descriptors
  PointCloudT::Ptr keypoints;                // filtered keypoints (same order)
};

// --- helpers ---
static bool has_suffix(const std::string& s, const std::string& suf) {
  if (s.size() < suf.size()) return false;
  for (size_t i = 0; i < suf.size(); ++i)
    if (std::tolower(s[s.size() - suf.size() + i]) != std::tolower(suf[i])) return false;
  return true;
}

PointCloudT::Ptr load_cloud_from_mesh(const std::string& path) {
  PointCloudT::Ptr cloud(new PointCloudT);
  if (has_suffix(path, ".ply")) {
    if (pcl::io::loadPLYFile<PointT>(path, *cloud) != 0)
      throw std::runtime_error("Failed to read PLY: " + path);
    return cloud;
  }
  if (has_suffix(path, ".obj")) {
    pcl::PolygonMesh mesh;
    if (pcl::io::loadOBJFile(path, mesh) != 0)
      throw std::runtime_error("Failed to read OBJ: " + path);
    pcl::fromPCLPointCloud2(mesh.cloud, *cloud);
    return cloud;
  }
  // fallback try PLY
  if (pcl::io::loadPLYFile<PointT>(path, *cloud) != 0)
    throw std::runtime_error("Unknown mesh format: " + path);
  return cloud;
}

PointCloudT::Ptr downsample(const PointCloudT::Ptr& cloud, float leaf) {
  if (leaf <= 0.f) return cloud;
  pcl::VoxelGrid<PointT> vg;
  vg.setLeafSize(leaf, leaf, leaf);
  vg.setInputCloud(cloud);
  PointCloudT::Ptr out(new PointCloudT);
  vg.filter(*out);
  return out;
}

// --- normals ---
NormalCloudT::Ptr compute_normals(const PointCloudT::Ptr& cloud, const YAML::Node& cfg) {
  int k = 20;
  if (cfg["normals"] && cfg["normals"]["k_search"])
    k = cfg["normals"]["k_search"].as<int>();
  pcl::NormalEstimationOMP<PointT, NormalT> ne;
  ne.setInputCloud(cloud);
  ne.setKSearch(k);
  using KdTree = pcl::search::KdTree<PointT>;
  KdTree::Ptr tree(new KdTree);
  ne.setSearchMethod(tree);
  NormalCloudT::Ptr normals(new NormalCloudT);
  ne.compute(*normals);
  return normals;
}

// --- keypoints ---
PointCloudT::Ptr detect_iss_keypoints(const PointCloudT::Ptr& cloud, const YAML::Node& cfg) {
  YAML::Node p = cfg["iss"];
  double salient = p && p["salient_radius"] ? p["salient_radius"].as<double>() : 0.06;
  double nonmax  = p && p["non_max_radius"] ? p["non_max_radius"].as<double>() : 0.04;
  double gamma21 = p && p["gamma_21"] ? p["gamma_21"].as<double>()
                                      : (p && p["threshold21"] ? p["threshold21"].as<double>() : 0.975);
  double gamma32 = p && p["gamma_32"] ? p["gamma_32"].as<double>()
                                      : (p && p["threshold32"] ? p["threshold32"].as<double>() : 0.975);
  int min_neighbors = p && p["min_neighbors"] ? p["min_neighbors"].as<int>() : 10;

  pcl::ISSKeypoint3D<PointT, PointT> iss;
  using KdTree = pcl::search::KdTree<PointT>;
  KdTree::Ptr tree(new KdTree);
  iss.setSearchMethod(tree);
  iss.setSalientRadius(salient);
  iss.setNonMaxRadius(nonmax);
  iss.setThreshold21(gamma21);
  iss.setThreshold32(gamma32);
  iss.setMinNeighbors(min_neighbors);
  iss.setNumberOfThreads(8);
  iss.setInputCloud(cloud);

  PointCloudT::Ptr keypoints(new PointCloudT);
  iss.compute(*keypoints);
  return keypoints;
}

PointCloudT::Ptr detect_harris_keypoints(const PointCloudT::Ptr& cloud,
                                         const NormalCloudT::Ptr& normals,
                                         const YAML::Node& cfg) {
  YAML::Node p = cfg["harris"];
  double radius    = p && p["radius"] ? p["radius"].as<double>() : 0.04;
  double threshold = p && p["threshold"] ? p["threshold"].as<double>() : 1e-6;
  bool   nonmax    = p && p["non_max_suppression"] ? p["non_max_suppression"].as<bool>() : true;
  std::string method = p && p["method"] ? p["method"].as<std::string>() : "HARRIS";

  pcl::HarrisKeypoint3D<PointT, pcl::PointXYZI> hk;
  hk.setInputCloud(cloud);
  hk.setNormals(normals);
  hk.setRadius(radius);
  hk.setThreshold(threshold);
  hk.setNonMaxSupression(nonmax);

  if (method == "NOBLE") hk.setMethod(pcl::HarrisKeypoint3D<PointT, pcl::PointXYZI>::NOBLE);
  else if (method == "LOWE") hk.setMethod(pcl::HarrisKeypoint3D<PointT, pcl::PointXYZI>::LOWE);
  else if (method == "TOMASI") hk.setMethod(pcl::HarrisKeypoint3D<PointT, pcl::PointXYZI>::TOMASI);
  else hk.setMethod(pcl::HarrisKeypoint3D<PointT, pcl::PointXYZI>::HARRIS);

  pcl::PointCloud<pcl::PointXYZI>::Ptr resp(new pcl::PointCloud<pcl::PointXYZI>);
  hk.compute(*resp);

  PointCloudT::Ptr keypoints(new PointCloudT);
  pcl::copyPointCloud(*resp, *keypoints);
  return keypoints;
}

// --- descriptors (compute at keypoints; filter NaNs; keep keypoints aligned) ---
DescAndKp<FPFHDescriptorT>
compute_fpfh_descriptors_with_kps(const PointCloudT::Ptr& surface,
                                  const PointCloudT::Ptr& keypoints,
                                  const NormalCloudT::Ptr& normals,
                                  double radius)
{
  FPFHCloudT::Ptr raw(new FPFHCloudT);
  pcl::FPFHEstimationOMP<PointT, NormalT, FPFHDescriptorT> est;
  est.setNumberOfThreads(8);
  est.setInputCloud(keypoints);
  est.setSearchSurface(surface);
  est.setInputNormals(normals);
  est.setRadiusSearch(radius);
  est.compute(*raw);

  DescAndKp<FPFHDescriptorT> out;
  out.desc.reset(new FPFHCloudT);
  out.keypoints.reset(new PointCloudT);
  out.desc->reserve(raw->size());
  out.keypoints->reserve(raw->size());
  for (std::size_t i = 0; i < raw->size(); ++i) {
    if (pcl::isFinite((*raw)[i])) {
      out.desc->push_back((*raw)[i]);
      out.keypoints->push_back(keypoints->points[i]);
    }
  }
  return out;
}

DescAndKp<SHOTDescriptorT>
compute_shot_descriptors_with_kps(const PointCloudT::Ptr& surface,
                                  const PointCloudT::Ptr& keypoints,
                                  const NormalCloudT::Ptr& normals,
                                  double radius)
{
  SHOTCloudT::Ptr raw(new SHOTCloudT);
  pcl::SHOTEstimationOMP<PointT, NormalT, SHOTDescriptorT> est;
  est.setNumberOfThreads(8);
  est.setSearchSurface(surface);
  est.setInputCloud(keypoints);
  est.setInputNormals(normals);
  est.setRadiusSearch(radius);
  est.compute(*raw);

  DescAndKp<SHOTDescriptorT> out;
  out.desc.reset(new SHOTCloudT);
  out.keypoints.reset(new PointCloudT);
  out.desc->reserve(raw->size());
  out.keypoints->reserve(raw->size());
  for (std::size_t i = 0; i < raw->size(); ++i) {
    if (pcl::isFinite((*raw)[i])) {
      out.desc->push_back((*raw)[i]);
      out.keypoints->push_back(keypoints->points[i]);
    }
  }
  return out;
}

// --- original-style visualization: shift cloud2 and draw green lines between matched keypoints
void visualize_original_style(const PointCloudT::Ptr& cloud1,
                              const PointCloudT::Ptr& cloud2,
                              const PointCloudT::Ptr& keypoints1,
                              const PointCloudT::Ptr& keypoints2,
                              const pcl::Correspondences& correspondences)
{
  pcl::visualization::PCLVisualizer viewer("Mesh Correspondences");
  viewer.setBackgroundColor(0.1, 0.1, 0.1);
  viewer.addCoordinateSystem(1.0);

  // cloud1 (white)
  viewer.addPointCloud(cloud1, "cloud1");

  // Shift cloud2 along +Y by 100.5 (same as your original snippet)
  Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
  transform(1, 3) = 100.5f;
  PointCloudT::Ptr cloud2_tf(new PointCloudT);
  pcl::transformPointCloud(*cloud2, *cloud2_tf, transform);
  pcl::visualization::PointCloudColorHandlerCustom<PointT> blue_color(cloud2_tf, 0, 150, 255);
  viewer.addPointCloud(cloud2_tf, blue_color, "cloud2");

  // lines between corresponding keypoints (use filtered, aligned keypoint sets!)
  for (std::size_t i = 0; i < correspondences.size(); ++i) {
    const auto& match = correspondences[i];
    const PointT& p1 = keypoints1->points[match.index_query];

    PointT p2 = keypoints2->points[match.index_match];
    p2.y += 100.5f; // apply same Y shift

    viewer.addLine<PointT, PointT>(p1, p2, 0, 255, 0, "line_" + std::to_string(i));
  }

  viewer.spin(); // blocks until 'q'
}

int main(int argc, char** argv) {
  if (argc < 6 || argc > 9) {
    std::cerr << "Usage: " << argv[0]
              << " mesh1.(ply|obj) mesh2.(ply|obj) <FPFH|SHOT> <ISS|HARRIS> config.yaml [leaf1] [leaf2] [viz]\n";
    return 1;
  }
  const std::string file1      = argv[1];
  const std::string file2      = argv[2];
  const std::string desc_type  = argv[3];
  const std::string kp_type    = argv[4];
  const std::string cfg_path   = argv[5];
  const bool do_viz            = (std::string(argv[argc-1]) == "viz");
  float leaf1 = (argc >= 7) ? std::stof(argv[6]) : 0.0f;
  float leaf2 = (argc >= 8 && std::string(argv[7]) != "viz") ? std::stof(argv[7]) : leaf1;

  YAML::Node cfg = YAML::LoadFile(cfg_path);

  try {
    // Load
    auto cloud1 = load_cloud_from_mesh(file1);
    auto cloud2 = load_cloud_from_mesh(file2);
    std::cout << "Loaded: " << file1 << " (" << cloud1->size() << " pts), "
              << file2 << " (" << cloud2->size() << " pts)\n";

    // Diagnostics like your original snippet (bounds)
    PointT min1, max1, min2, max2;
    pcl::getMinMax3D(*cloud1, min1, max1);
    pcl::getMinMax3D(*cloud2, min2, max2);
    std::cout << "--> Mesh 1 dims (W x H x D): "
              << (max1.x - min1.x) << " x " << (max1.y - min1.y) << " x " << (max1.z - min1.z) << "\n";
    std::cout << "--> Mesh 2 dims (W x H x D): "
              << (max2.x - min2.x) << " x " << (max2.y - min2.y) << " x " << (max2.z - min2.z) << "\n";

    // Downsample
    cloud1 = downsample(cloud1, leaf1);
    cloud2 = downsample(cloud2, leaf2);
    std::cout << "Downsampled: cloud1=" << cloud1->size() << ", cloud2=" << cloud2->size() << "\n";

    // Normals
    auto normals1 = compute_normals(cloud1, cfg);
    auto normals2 = compute_normals(cloud2, cfg);
    std::cout << "Computed normals.\n";

    // Keypoints
    PointCloudT::Ptr kps1, kps2;
    if (kp_type == "ISS") {
      kps1 = detect_iss_keypoints(cloud1, cfg);
      kps2 = detect_iss_keypoints(cloud2, cfg);
    } else if (kp_type == "HARRIS") {
      kps1 = detect_harris_keypoints(cloud1, normals1, cfg);
      kps2 = detect_harris_keypoints(cloud2, normals2, cfg);
    } else {
      throw std::runtime_error("Unknown keypoint type: " + kp_type);
    }
    std::cout << "Keypoints: cloud1=" << kps1->size() << ", cloud2=" << kps2->size() << "\n";

    // Descriptor params (from YAML or defaults similar to your original)
    double fpfh_radius = (cfg["fpfh"] && cfg["fpfh"]["radius_search"]) ? cfg["fpfh"]["radius_search"].as<double>() : 0.08;
    double shot_radius = (cfg["shot"] && cfg["shot"]["radius_search"]) ? cfg["shot"]["radius_search"].as<double>() : 0.08;

    // Descriptors & correspondences (and keep filtered keypoints for viz)
    pcl::CorrespondencesPtr corr(new pcl::Correspondences);

    if (desc_type == "FPFH") {
      auto d1 = compute_fpfh_descriptors_with_kps(cloud1, kps1, normals1, fpfh_radius);
      auto d2 = compute_fpfh_descriptors_with_kps(cloud2, kps2, normals2, fpfh_radius);
      std::cout << "FPFH sizes (post-filter): " << d1.desc->size() << " vs " << d2.desc->size() << "\n";

      pcl::registration::CorrespondenceEstimation<FPFHDescriptorT, FPFHDescriptorT> est;
      est.setInputSource(d1.desc);
      est.setInputTarget(d2.desc);
      est.determineReciprocalCorrespondences(*corr);
      std::cout << "Reciprocal correspondences (one-to-one): " << corr->size() << "\n";

      if (do_viz) {
        visualize_original_style(cloud1, cloud2, d1.keypoints, d2.keypoints, *corr);
      }

    } else if (desc_type == "SHOT") {
      auto d1 = compute_shot_descriptors_with_kps(cloud1, kps1, normals1, shot_radius);
      auto d2 = compute_shot_descriptors_with_kps(cloud2, kps2, normals2, shot_radius);
      std::cout << "SHOT sizes (post-filter): " << d1.desc->size() << " vs " << d2.desc->size() << "\n";

      pcl::registration::CorrespondenceEstimation<SHOTDescriptorT, SHOTDescriptorT> est;
      est.setInputSource(d1.desc);
      est.setInputTarget(d2.desc);
      est.determineReciprocalCorrespondences(*corr);
      std::cout << "Reciprocal correspondences (one-to-one): " << corr->size() << "\n";

      if (do_viz) {
        visualize_original_style(cloud1, cloud2, d1.keypoints, d2.keypoints, *corr);
      }

    } else {
      throw std::runtime_error("Unknown descriptor type: " + desc_type);
    }

    std::cout << "Done.\n";
    return 0;

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 2;
  }
}
