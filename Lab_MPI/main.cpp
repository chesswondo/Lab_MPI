#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <cstdint>
#include <chrono>
#include <algorithm>
#include <string>
#include <windows.h> // For directory traversal
#include <direct.h>  // For _mkdir
#include <locale>
#include <codecvt>
#include <mpi.h>     // MPI header

// Function declaration
std::string wcharToString(const wchar_t* wstr);

#pragma pack(push, 1)
struct BMPHeader {
    char sign[2];
    int size;
    int reserved;
    int dataOffset;
    int headerSize;
    int width;
    int height;
    short planes;
    short bitPerPixel;
    int compression;
    int imageSize;
    int xPixelsPerM;
    int yPixelsPerM;
    int colorsUsed;
    int importantColors;
};
#pragma pack(pop)

inline int clamp(int value, int minVal, int maxVal) {
    return (value < minVal) ? minVal : (value > maxVal) ? maxVal : value;
}

class GaussianFilter {
public:
    GaussianFilter(int radius) : radius(radius) {
        generateKernel();
    }

    void applyFilter(std::vector<std::vector<uint8_t>>& channel) {
        int width = channel[0].size();
        int height = channel.size();
        std::vector<std::vector<uint8_t>> temp(height, std::vector<uint8_t>(width));

        // Horizontal pass
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                double sum = 0, weightSum = 0;
                for (int k = -radius; k <= radius; k++) {
                    int xIndex = clamp(x + k, 0, width - 1);
                    sum += channel[y][xIndex] * kernel[k + radius];
                    weightSum += kernel[k + radius];
                }
                temp[y][x] = static_cast<uint8_t>(sum / weightSum);
            }
        }

        // Vertical pass
        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                double sum = 0, weightSum = 0;
                for (int k = -radius; k <= radius; k++) {
                    int yIndex = clamp(y + k, 0, height - 1);
                    sum += temp[yIndex][x] * kernel[k + radius];
                    weightSum += kernel[k + radius];
                }
                channel[y][x] = static_cast<uint8_t>(sum / weightSum);
            }
        }
    }

private:
    int radius;
    std::vector<double> kernel;

    void generateKernel() {
        double sigma = radius / 2.0;
        double sum = 0;
        kernel.resize(2 * radius + 1);

        for (int i = -radius; i <= radius; i++) {
            kernel[i + radius] = exp(-0.5 * (i * i) / (sigma * sigma));
            sum += kernel[i + radius];
        }

        for (double& val : kernel) val /= sum;
    }
};

void processImage(const std::string& inputPath, const std::string& outputPath, int radius) {
    std::ifstream file(inputPath, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file: " << inputPath << std::endl;
        return;
    }

    BMPHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(BMPHeader));
    if (header.bitPerPixel != 24) {
        std::cerr << "Only 24-bit BMP files are supported!" << std::endl;
        return;
    }

    int width = header.width;
    int height = header.height;
    file.seekg(header.dataOffset, std::ios::beg);

    std::vector<std::vector<uint8_t>> red(height, std::vector<uint8_t>(width));
    std::vector<std::vector<uint8_t>> green(height, std::vector<uint8_t>(width));
    std::vector<std::vector<uint8_t>> blue(height, std::vector<uint8_t>(width));

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            file.read(reinterpret_cast<char*>(&blue[i][j]), 1);
            file.read(reinterpret_cast<char*>(&green[i][j]), 1);
            file.read(reinterpret_cast<char*>(&red[i][j]), 1);
        }
    }
    file.close();

    GaussianFilter filter(radius);
    auto start = std::chrono::high_resolution_clock::now();

    filter.applyFilter(red);
    filter.applyFilter(green);
    filter.applyFilter(blue);

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Processed: " << inputPath << " in "
        << std::chrono::duration<double, std::milli>(end - start).count() << " ms" << std::endl;

    std::ofstream outFile(outputPath, std::ios::binary);
    outFile.write(reinterpret_cast<char*>(&header), sizeof(BMPHeader));
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            outFile.write(reinterpret_cast<char*>(&blue[i][j]), 1);
            outFile.write(reinterpret_cast<char*>(&green[i][j]), 1);
            outFile.write(reinterpret_cast<char*>(&red[i][j]), 1);
        }
    }
    outFile.close();
}

std::vector<std::string> getBmpFilesInDirectory(const std::string& directory) {
    std::vector<std::string> imageFiles;
    WIN32_FIND_DATA findFileData;

    // Convert directory path to wide-character string
    std::wstring wideDirectory = std::wstring(directory.begin(), directory.end()) + L"/*.bmp";
    HANDLE hFind = FindFirstFile(wideDirectory.c_str(), &findFileData);

    if (hFind == INVALID_HANDLE_VALUE) {
        std::cerr << "Error opening directory: " << directory << std::endl;
        return imageFiles;
    }

    do {
        if (!(findFileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
            // Convert wide-character file name to std::string
            imageFiles.push_back(wcharToString(findFileData.cFileName));
        }
    } while (FindNextFile(hFind, &findFileData) != 0);

    FindClose(hFind);
    return imageFiles;
}

std::string wcharToString(const wchar_t* wstr) {
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    return converter.to_bytes(wstr);
}

void createDirectoryIfNotExists(const std::string& path) {
    if (_mkdir(path.c_str())) {
        if (errno != EEXIST) {
            std::cerr << "Error creating directory: " << path << std::endl;
        }
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv); // Initialize MPI

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get process rank
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Get total number of processes
    std::cout << "Process " << rank << " of " << size << " is running." << std::endl;

    if (argc < 3) {
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0] << " <radius> <parallelization_mode>" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    int radius = std::stoi(argv[1]);

    // Ensure output directory exists (only rank 0 creates it)
    if (rank == 0) {
        createDirectoryIfNotExists("processed");
    }
    MPI_Barrier(MPI_COMM_WORLD); // Synchronize all processes

    // Get list of BMP files in the input directory (only rank 0 reads the directory)
    std::vector<std::string> imageFiles;
    if (rank == 0) {
        imageFiles = getBmpFilesInDirectory("images");
    }

    // Broadcast the number of images to all processes
    int numImages = imageFiles.size();
    MPI_Bcast(&numImages, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Distribute image files to all processes
    std::vector<std::string> localImageFiles;
    if (rank == 0) {
        // Rank 0 distributes the work
        for (int i = 0; i < numImages; i++) {
            int targetRank = i % size;
            if (targetRank == 0) {
                localImageFiles.push_back(imageFiles[i]);
            }
            else {
                MPI_Send(imageFiles[i].c_str(), imageFiles[i].size() + 1, MPI_CHAR, targetRank, 0, MPI_COMM_WORLD);
            }
        }
    }
    else {
        // Other processes receive their assigned files
        char buffer[256];
        MPI_Recv(buffer, 256, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        localImageFiles.push_back(buffer);
    }

    // Process assigned images
    auto start_total = std::chrono::high_resolution_clock::now();
    for (const auto& file : localImageFiles) {
        processImage("images/" + file, "processed/" + file, radius);
    }
    auto end_total = std::chrono::high_resolution_clock::now();

    // Print timing information
    if (rank == 0) {
        std::cout << "Total time: " << std::chrono::duration<double, std::milli>(end_total - start_total).count() << " ms" << std::endl;
    }

    MPI_Finalize(); // Finalize MPI
    return 0;
}