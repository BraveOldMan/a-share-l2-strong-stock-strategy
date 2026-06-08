import os
import shutil
import glob

# 清理模型目录内白名单之外的文件
model_dir = 'models'
keep_set = {'daily_cache', '_polars_temp', 'ecosystem_features.parquet', 'cross_section_features.parquet'}

if os.path.exists(model_dir):
    deleted = []
    for f in os.listdir(model_dir):
        if f not in keep_set:
            file_path = os.path.join(model_dir, f)
            if os.path.isfile(file_path):
                os.remove(file_path)
                deleted.append(f)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
                deleted.append(f)
    print("已清理的历史日志/模型缓存文件:\n" + "\n".join(f" - {d}" for d in deleted))
else:
    print("无需清理，目录不存在")

# 清理根目录残留的临时垃圾文件夹
def clean_root_temp():
    temp_dirs = glob.glob("temp_buffer_*") + glob.glob("temp_tick_*")
    if temp_dirs:
        for t in temp_dirs:
            if os.path.isdir(t):
                shutil.rmtree(t, ignore_errors=True)
                print(f"已清理残留临时目录: {t}")

clean_root_temp()
