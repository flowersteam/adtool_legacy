from leaf.tests.test_integration_leaf import DiskPipeline, DiskLocator
import os
if __name__ == "__main__":
    a = DiskPipeline()
    os.mkdir("/tmp/PipelineDir")
    os.mkdir("/tmp/ResDir")
    a.save_leaf()
    uid = a.uid
