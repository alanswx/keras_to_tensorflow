# keras_to_tensorflow
Convert keras models to tensorflow frozen graph for use on cell phones, etc

The last parameter of the script takes the path to the freeze graph tool. Build it here:
```
  bazel build tensorflow/python/tools:freeze_graph
```
It usually lives here off of your tensorflow directory:
```
tensorflow/bazel-bin/tensorflow/python/tools/freeze_graph
```

The script needs to be fixed so it doesn't put ./ in front of the paths. I was having a little trouble getting things to work. Feel free to fix and submit a pull request.
