import h5py
import numpy as np
import os


def average_data(algorithm="", dataset="", goal="", times=10):
    test_acc = get_all_results_for_one_algo(algorithm, dataset, goal, times)

    max_accuracy = []
    for i in range(times):
        max_accuracy.append(test_acc[i].max())

    print("std for best accuracy:", np.std(max_accuracy))
    print("mean for best accuracy:", np.mean(max_accuracy))


def get_all_results_for_one_algo(algorithm="", dataset="", goal="", times=10):
    test_acc = []
    algorithms_list = [algorithm] * times
    for i in range(times):
        file_name = dataset + "_" + algorithms_list[i] + "_" + goal + "_" + str(i)
        test_acc.append(np.array(read_data_then_delete(file_name, delete=False)))

    return test_acc


def read_data_then_delete(file_name, delete=False):
    # Get the directory where this script is located (system/utils/)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up to system/, then to results/
    results_base = os.path.join(os.path.dirname(current_dir), "results")
    
    # 文件保存在子目录中: system/results/{dataset}_{algorithm}_{goal}/
    # file_name 格式: {dataset}_{algorithm}_{goal}_{times}
    # 需要提取前三部分作为子目录名
    file_name_parts = file_name.rsplit('_', 1)  # 分离出最后的 times
    if len(file_name_parts) == 2:
        folder_name = file_name_parts[0]  # 不包含times的部分作为文件夹名
        file_path = os.path.join(results_base, folder_name, file_name + ".h5")
    else:
        file_path = os.path.join(results_base, file_name + ".h5")
    
    # Check if h5 file exists, otherwise try npy format
    if not os.path.exists(file_path):
        # 尝试npy格式
        npy_folder = file_name.rsplit('_', 1)[0]
        npy_path = os.path.join(results_base, npy_folder, npy_folder + "_test_acc.npy")
        if os.path.exists(npy_path):
            print(f"Note: Reading from npy format: {npy_path}")
            rs_test_acc = np.load(npy_path)
            if delete:
                os.remove(npy_path)
            print("Length: ", len(rs_test_acc))
            return rs_test_acc
        else:
            # List available files to help debugging
            results_dir = results_base
            if os.path.exists(results_dir):
                available_files = os.listdir(results_dir)
                error_msg = f"\n[ERROR] Cannot find result files for: {file_name}\n"
                error_msg += f"  Expected: {file_path}\n"
                error_msg += f"  Or: {npy_path}\n\n"
                error_msg += f"Available files in {results_dir}:\n"
                if available_files:
                    for f in available_files:
                        error_msg += f"  - {f}\n"
                else:
                    error_msg += "  (directory is empty)\n\n"
                error_msg += "\nPossible reasons:\n"
                error_msg += "  1. Training completed but save_results() was not called\n"
                error_msg += "  2. No evaluation was performed (check eval_gap setting)\n"
                error_msg += "  3. Training failed before evaluation\n"
                error_msg += "  4. Check training logs for 'WARNING: No evaluation results to save!'\n"
            else:
                error_msg = f"\n[ERROR] Results directory does not exist: {results_dir}\n"
                error_msg += "Training may have failed before saving any results.\n"
            raise FileNotFoundError(error_msg)

    with h5py.File(file_path, 'r') as hf:
        rs_test_acc = np.array(hf.get('rs_test_acc'))

    if delete:
        os.remove(file_path)
    print("Length: ", len(rs_test_acc))

    return rs_test_acc