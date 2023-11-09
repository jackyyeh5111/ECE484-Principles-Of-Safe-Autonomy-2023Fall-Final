## Execution

1. Download [frames](https://uofi.box.com/s/82lk65dg8a9vkvc4hn17ffag5car7dva).

2. Execute commands
    ```
    $ cd ECE484-Principles-Of-Safe-Autonomy-2023Fall-Final/f1tenth_ros1_ws/src/test_scripts

    $ python3 test.py -p 121,470,312,0 \
                      --sat_thresh 60,255 \
                      --val_thresh 50,255 \
                      --hue_thresh 10,40 \
                      --gradient_thresh 15,30 \
                      --vo
    ```