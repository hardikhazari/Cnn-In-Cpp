// ============================================================================
// Dataset.h — Abstract base class for datasets.
//
// A Dataset provides a way to access individual data samples (features + labels)
// by index. It does NOT handle batching, shuffling, or asynchronous loading;
// that is the job of the DataLoader.
//
// Think of this as a simple array-like interface:
//   auto [image, label] = dataset.get_item(i);
// ============================================================================
#pragma once
#include <string>
#include <fstream>
#include <stdexcept>
#include <charconv>
#include <iostream>
#include "../core/Tensor.h"

namespace CnnInCpp {

class Dataset {
public:
    Tensor images, labels;

    static inline void skipComments(std::ifstream& f) {
        char c; f>>c;
        while (c=='#') { f.ignore(256,'\n'); f>>c; }
        f.unget();
    }

    static inline Tensor load_ppm_grayscale(const std::string& fp) {
        std::ifstream f(fp, std::ios::binary);
        if (!f) throw std::runtime_error("Cannot open: "+fp);
        std::string m; f>>m;
        if (m!="P6"&&m!="P3") throw std::runtime_error("Bad PPM: "+m);
        skipComments(f);
        int W,H,mc; f>>W>>H>>mc; f.get();
        Tensor img(1,1,H,W);
        float* d=img.data.data();
        for (int y=0;y<H;++y) for (int x=0;x<W;++x) {
            float r,g,b;
            if (m=="P6"){unsigned char px[3];f.read((char*)px,3);r=px[0];g=px[1];b=px[2];}
            else{int ri,gi,bi;f>>ri>>gi>>bi;r=ri;g=gi;b=bi;}
            d[y*W+x]=0.299f*r+0.587f*g+0.114f*b;
        }
        return img;
    }

    static inline Tensor load_ppm_rgb(const std::string& fp) {
        std::ifstream f(fp, std::ios::binary);
        if (!f) throw std::runtime_error("Cannot open: "+fp);
        std::string m; f>>m;
        if (m!="P6"&&m!="P3") throw std::runtime_error("Bad PPM: "+m);
        skipComments(f);
        int W,H,mc; f>>W>>H>>mc; f.get();
        Tensor img(1,3,H,W);
        float* r_=img.data.data();
        float* g_=r_+H*W;
        float* b_=g_+H*W;
        for (int y=0;y<H;++y) for (int x=0;x<W;++x) {
            float r,g,b;
            if (m=="P6"){unsigned char px[3];f.read((char*)px,3);r=px[0];g=px[1];b=px[2];}
            else{int ri,gi,bi;f>>ri>>gi>>bi;r=ri;g=gi;b=bi;}
            r_[y*W+x]=r; g_[y*W+x]=g; b_[y*W+x]=b;
        }
        return img;
    }

    inline void normalize() {
        float* d=images.data.data();
        #pragma omp simd
        for (int i=0;i<images.size();++i) d[i]/=255.0f;
    }

    static inline Dataset load_csv(const std::string& path, bool has_header = false, int num_classes = 10, std::vector<int> image_shape = {}, int manual_label_col = -1) {
        std::ifstream file(path);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open CSV file: " + path);
        }

        std::string line;
        line.reserve(8192);

        if (has_header) {
            std::getline(file, line);
        }

        // ==========================================
        // PASS 1: Metadata Scouting
        // ==========================================
        int num_samples = 0;
        int total_columns = -1;
        int max_col_0 = -1, max_col_last = -1;

        while (std::getline(file, line)) {
            if (line.empty()) continue;
            num_samples++;
            
            if (total_columns == -1) {
                total_columns = 1;
                for (char c : line) {
                    if (c == ',') total_columns++;
                }
            }

            if (num_samples <= 50 && manual_label_col == -1) {
                size_t first_comma = line.find(',');
                size_t last_comma = line.find_last_of(',');
                if (first_comma != std::string::npos) {
                    int v0 = -1;
                    std::from_chars(line.data(), line.data() + first_comma, v0);
                    max_col_0 = std::max(max_col_0, v0);

                    int vlast = -1;
                    std::from_chars(line.data() + last_comma + 1, line.data() + line.length(), vlast);
                    max_col_last = std::max(max_col_last, vlast);
                }
            }
        }

        int num_features = total_columns - 1;
        int label_col = manual_label_col;

        if (label_col == -1) {
            if (max_col_0 < num_classes && max_col_last >= num_classes) {
                label_col = 0;
            } else if (max_col_last < num_classes && max_col_0 >= num_classes) {
                label_col = num_features; // The last valid column
            } else {
                throw std::runtime_error("CSV Format Ambiguity: Cannot safely auto-detect label. Please explicitly pass manual_label_col (e.g., 0 for first column).");
            }
        }

        // ==========================================
        // PASS 2: Data Extraction
        // ==========================================
        file.clear();
        file.seekg(0, std::ios::beg);
        if (has_header) std::getline(file, line);

        Dataset ds;
        int c = 1, h = 1, w = num_features;
        if (!image_shape.empty()) {
            if (image_shape.size() != 3) {
                throw std::runtime_error("Dataset: image_shape must be {C, H, W}");
            }
            if (image_shape[0] * image_shape[1] * image_shape[2] != num_features) {
                throw std::runtime_error("Dataset: image_shape product does not match inferred num_features.");
            }
            c = image_shape[0]; 
            h = image_shape[1]; 
            w = image_shape[2];
        }

        ds.images = Tensor(num_samples, c, h, w);
        ds.labels = Tensor(num_samples, num_classes);
        ds.labels.fill(0.0f);

        float* img_data = ds.images.data.data();
        float* lbl_data = ds.labels.data.data();
        int img_offset = 0;
        int lbl_offset = 0;
        int row_idx = 0;

        while (std::getline(file, line)) {
            if (line.empty()) continue;
            
            size_t start = 0;
            size_t end = line.find(',');
            
            int col_idx = 0;
            int label = -1;
            int p = 0;

            while (start < line.length()) {
                if (end == std::string::npos) end = line.length();
                
                if (col_idx == label_col) {
                    std::from_chars(line.data() + start, line.data() + end, label);
                    if (label >= 0 && label < num_classes) {
                        lbl_data[lbl_offset + label] = 1.0f;
                    }
                    if (row_idx == 0) {
                        std::cout << "[DEBUG] Auto-Detect Resolved: Label is Column " << label_col << "\n";
                        std::cout << "[DEBUG] First CSV Row -> Parsed Label: " << label << " | One-Hot: [";
                        for (int cl = 0; cl < num_classes; ++cl) {
                            std::cout << lbl_data[lbl_offset + cl] << (cl == num_classes - 1 ? "" : ", ");
                        }
                        std::cout << "]\n";
                    }
                } else if (p < num_features) {
                    float val = 0.0f;
                    std::from_chars(line.data() + start, line.data() + end, val);
                    img_data[img_offset++] = val;
                    p++;
                }
                
                col_idx++;
                start = end + 1;
                end = line.find(',', start);
            }
            lbl_offset += num_classes;
            row_idx++;
        }
        return ds;
    }
    
    // Legacy overload for backward compatibility with pre-existing caller logic
    static inline Dataset load_csv(const std::string& path, int num_samples, int num_pixels, bool has_header = false, int num_classes = 10, int manual_label_col = -1) {
        std::vector<int> shape;
        if (num_pixels == 784) shape = {1, 28, 28};
        else if (num_pixels == 3072) shape = {3, 32, 32};
        else shape = {1, 1, num_pixels};
        return load_csv(path, has_header, num_classes, shape, manual_label_col);
    }
};

} // namespace CnnInCpp
