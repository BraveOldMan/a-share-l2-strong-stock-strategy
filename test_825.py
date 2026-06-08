import sys
import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()
    print('Testing 20250825..', flush=True)
    try:
        import batch_pipeline
        df = batch_pipeline.process_single_day('20250825')
        print('Shape:', df.shape, flush=True)
    except Exception as e:
        print('Error:', e, flush=True)
