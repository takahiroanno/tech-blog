---
layout: post
title:  "tensorflow servingの話 その1"
date:   2017-12-20 00:00:00 +0900
---

## まずはじめに

機械学習を利用したアプリケーションの乘ったサーバー、運用がめちゃくちゃ面倒くさいですね。tensorflowのコードとそれ以外の部分がゴチャゴチャしてきてしまったり、CPUやメモリのリソースがかなり必要だったり等々…

その中でも特に面倒なのが、学習したモデルファイルをサーバーに配布して、サービスを止めることなく入れ換えることです。
そこで今回は、この一番面倒な部分をどうこうすることなく、良い塩梅にモデルの更新ができる [tensorflow serving][tensorflow-serving] (以下serving)について数回にわけて実際の運用までを紹介していきます。

## servingとは

ザックリ言うとtensorflowの`GraphDef`を使って、`gprc`でリクエストを受け付けてsess.runしてくれるめっちゃ速いC++で書かれたサーバーです。実際はもっと広範な用途まで想定されています、その辺は公式ドキュメントの[architecture overview][architecture-overview] を。

servingは、tensorflowにおけるgraphの様な何か計算をするservableなオブジェクトが複数読み込みます。読み込みはデフォルトではローカルのファイルパスが指定できる仕組みになっており、ユーザーは指定したディレクトリ以下にモデルファイルをexportするだけでservingに新しいモデルを読み込ませることができます。

ここで大事なのは、servingは`GraphDef`と`SignatureDef`を使って呼び出せるモデル、関数をビルドするので、serving側には学習済みモデル以外のコードが一切必要ないということです。(便利ポイント！！！)

servingはgrpcしか受け付けないのでwebアプリケーションなどから利用する場合、手前に何かサーバーを立てる必要がありますが、前述の仕組みにより、手前のサーバーは機械学習の入出力さえ知っていれば中のモデルがどういう形になっているか知らなくて済みます。

## まずservingを動かす

前置きはこのくらいにして早速servingを起動しましょう。公式にはdockerhubにコンテナがないので、手元でコンテナを作ります。
```docker
FROM ubuntu:16.04

RUN apt-get update && \
    apt-get upgrade -qy && \
    apt-get install -y --no-install-recommends \
      ca-certificates curl && \
    echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | tee /etc/apt/sources.list.d/tensorflow-serving.list && \
    curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | apt-key add - && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      tensorflow-model-server && \
    apt-get clean && \
    rm -rf /var/cache/apt/archives/* /var/lib/apt/lists/*

EXPOSE 8500
```
こんな感じで`tensorflow_model_server`コマンドの使えるコンテナのイメージのDockerfileができます。

イメージをビルドし
```bash
docker build -t serving-example .
```

コンテナを起動します
```bash
docker run --name serving-example -v `pwd`:/root/serving-example -p 8500:8500 -it serving-example /bin/bash
```

そしてコンテナ内でservingを起動します
```bash
tensorflow_model_server --model_name='default' --model_base_path=/root/serving-example/tmp
```

`/root/serving-example`にはこの今いる場所をmountしたのでここにtmpディレクトリを作り、その中にモデルファイルを置くことでservingにモデルをロードさせられます。今はまだ何も置いていないので`No versions of servable default found under base path`と怒られているはずです。

## graphを作る

何はともあれgraphを作りましょう。
```python
# define graph
graph = tf.Graph()
with graph.as_default():
    x = tf.placeholder(dtype=tf.int64, shape=(), name='x')
    y = tf.placeholder(dtype=tf.int64, shape=(), name='y')
    x_add_y = tf.add(x=x, y=y)

    # test run graph
    with tf.Session() as sess:
        print('local x_add_y run result: {}'.format(sess.run(x_add_y, feed_dict={x: X, y: Y})))
```
`x + y`です。XとYに適当な値を入れて呼ぶと実行されます。

次にこれをserving用にexportします。
```python
# save current graph for serving
builder = tf.saved_model.builder.SavedModelBuilder(EXPORT_DIR)
with tf.Session(graph=graph) as sess:
    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            'x_add_y': tf.saved_model.signature_def_utils.build_signature_def(
                inputs={'x': tf.saved_model.utils.build_tensor_info(x), 'y': tf.saved_model.utils.build_tensor_info(y)},
                outputs={'x_add_y':  tf.saved_model.utils.build_tensor_info(x_add_y)},
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME,
            ),
        },
    )
    builder.save()
```
`SavedModelBuilder`は変数とgraphをSavedModelのprotocol-buffer形式で保存してくれます。
このbuilderに対してメタ情報として保存したい変数とgraphのあるsessionや`signature_def_map`を渡し、saveするという流れになります。今回はただの足し算で、変数の値は必要なくgraphのみで良いので、sess.runしたのとは別のsessionで保存しています。

`signature_def_map`は1つのgraphに対して、実行したい計算が通常複数あるので、それに対応するため名前とsignature_defのdictになっています。

`signature_def`はというと、実際に計算を行ない結果を取り出すために必要な情報が格納されています。`inputs`と`outputs`はそれぞれ、計算の入力と出力に必要な変数名とそれを入れるべき`tensor_info`のdictです。ここでの名前はgraph内での名前(placeholderのname引数)と一致している必要はありません。graphの方で名前を変えてしまっても、ここを統一しておけば同じinterfaceで呼び出せるというわけです。

この辺は`tf.saved_model.utils`以下に便利な各種builderが用意されているのでこれらを使うのが良いでしょう。

EXPORT_DIRはコンテナをマウントしたディレクトリから`tmp/<version:int>`となるようなパスを指定してください。

さて、ここまで実行してからmodel_serverの出力を見ると
```bash
Loading SavedModel from: /root/serving-example/tmp/2
Restoring SavedModel bundle.
The specified SavedModel has no variables; no checkpoints were restored.
Running LegacyInitOp on SavedModel bundle.
Loading SavedModel: success. Took 3604 microseconds.
Successfully loaded servable version {name: default version: 2}
```
というような感じのログが流れているのではないでしょうか。
ここからさらにversionを増やして再度実行すると、その都度最新のモデルが読み込まれ
```bash
Quiescing servable version {name: default version: 1}
Done quiescing servable version {name: default version: 1}
Unloading servable version {name: default version: 1}
Calling MallocExtension_ReleaseToSystem() after servable unload with 534
Done unloading servable version {name: default version: 1}
```
という風に最新バージョンのロードに成功した後に古いモデルをアンロードしてくれます。

さらに、なんとservingはディレクトリを削除を検知して、それが現在メモリに載っているモデルの場合自動で古いモデルを読み込みます。

更新もロールバックもファイルの移動だけで済んでしまうわけです。

## servingにリクエストを投げてみる

gprcは`pip install grpcio`で入るのですが、servingに投げるリクエストを作るための`tensorflow-serving-api`パッケージはpython3向けのものがありません。が、python2向けのものがそのままptyhon3でも動くので、 [pip][pip] からzipを落としてきてローカルに置きましょう。([issue][issue]はあるのですが、ここはあんまり対応する気がなさそうなので気長に待ちましょう)

まずクライアントのstubを作成します
```python
from grpc.beta import implementations
from tensorflow_serving.apis import prediction_service_pb2

# create grpc stub
channel = implementations.insecure_channel(SERVING_HOST, SERVING_PORT)
stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
```

次に投げるリクエストを作成します
```python
from tensorflow.core.framework import types_pb2
from google.protobuf import wrappers_pb2
from tensorflow_serving.apis import predict_pb2

# create predict request
request = predict_pb2.PredictRequest()
request.model_spec.name = MODEL_NAME
version = wrappers_pb2.Int64Value()
version.value = VERSION
request.model_spec.version.CopyFrom(version)
request.model_spec.signature_name = 'x_add_y'

request.inputs['x'].dtype = types_pb2.DT_INT64
request.inputs['x'].int64_val.append(X)
request.inputs['y'].dtype = types_pb2.DT_INT64
request.inputs['y'].int64_val.append(Y)
```
リクエストにはいくつか種類がありますが、今回は`predict`を使います。servingに投げるリクエストは、`model_spec`下にあるリクエストを投げる対象のモデルと`inputs`下にある入力内容の2つの要素からなります。

`model_spec`で指定できる内容はモデルの`name`、そして`version`、`signature_name`です。が、なんとバージョンは指定する必要がなく、その場合自動で最新のバージョンが呼ばれます。呼び出し側は今どのバージョンがロードされているか意識する必要がないわけです。

`inputs`は`signature_def_map`で指定した変数名とその中身というdictになります。placeholderで指定しておいたdtypeと同じ型を明示し、それに対応したプロパティに値をappendしてやることで入力値をセットできます。

これで準備はOKです。リクエストを投げましょう。
```python
result_future = stub.Predict.future(request, 1)
result = result_future.result()
print('serving x_add_y run result: {}'.format(result.outputs['x_add_y'].int64_val[0]))
```
上手く値が出力されたでしょうか。
`outputs`はinputsと同じく変数名をkeyとするdictになっており、graph内でのdtypeと同じプロパティにアクセスすることで中身が取得できます。こちらもinputsと同じく配列になっていますが、今回は出力は1つだけです。

## まとめ

生tensorflowでローカルにserving用のモデルを出力し、実際にリクエストを投げるところまでできました。

これで自分で定義したグラフをservingでバージョニングすることができるようになりました！サーバーを再起動しなくてもモデルの更新ができる！……のですが、これだと各servingサーバーのローカルにモデルファイルを配布する必要があり、pythonで全てを書いていた時と比較して運用が楽になっていません。そもそも生tensorflowを書くのがつらすぎる…

ということで、タイトルにその1とある通り、servingの話はまだ続きます。次回以降、ローカルのファイルシステムではなくS3でモデルファイルを管理し、生tensorflowを書くのをやめてkerasやestimatorを使い、実際のAWS上でのインフラ構成を紹介します。乞う御期待。

[tensorflow-serving]: https://www.tensorflow.org/serving/
[architecture-overview]: https://www.tensorflow.org/serving/architecture_overview
[mnist-example]: https://www.tensorflow.org/serving/serving_basic
[pip]: https://pypi.python.org/pypi/tensorflow-serving-api/1.4.0
[issue]: https://github.com/tensorflow/serving/issues/581
