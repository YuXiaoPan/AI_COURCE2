package me.peyppicp

import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.BackpropType
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.conf.layers.GravesLSTM
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.api.buffer.util.DataTypeUtil
import org.nd4j.linalg.lossfunctions.LossFunctions
import java.util.*

fun main(args: Array<String>) {

    DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE)

    val filePath = "F:\\WorkSpace\\idea project location\\AI-Emoji\\src\\main\\resources\\standardData.txt"
    val lstmLayerSize = 200
    val miniBatchSize = 40
    val exampleLength = 1000
    val tbpttLength = 50                       //Length for truncated backpropagation through time. i.e., do parameter updates ever 50 characters
    val numEpochs = 1                            //Total number of training epochs
    val generateSamplesEveryNMinibatches = 10  //How frequently to generate samples from the network? 1000 characters / 50 tbptt length: 20 parameter updates per minibatch
    val nSamplesToGenerate = 4                    //Number of samples to generate after each training epoch
    val nCharactersToSample = 300
    val rng = Random(12345)
    val generationInitialization: String? = null

    val dataSetIterator = EmojiDataSetIterator(filePath, Charsets.UTF_8, miniBatchSize, exampleLength, rng)

    val nOut = dataSetIterator.totalOutcomes()

    val conf = NeuralNetConfiguration.Builder()
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
            .learningRate(0.01)
            .seed(12345)
            .regularization(true)
            .l2(0.1)
            .weightInit(WeightInit.XAVIER)
            .updater(Updater.RMSPROP)
            .list()
            .layer(0, GravesLSTM.Builder().nIn(dataSetIterator.inputColumns()).nOut(lstmLayerSize)
                    .activation(Activation.SOFTMAX).build())
            .layer(1, GravesLSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize)
                    .activation(Activation.SOFTMAX).build())
            .layer(2, RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT).activation(Activation.SOFTMAX)        //MCXENT + softmax for classification
                    .nIn(lstmLayerSize).nOut(nOut).build())
            .backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(tbpttLength).tBPTTBackwardLength(tbpttLength)
            .pretrain(false).backprop(true)
            .build()

    val net = MultiLayerNetwork(conf)
    net.init()
    net.setListeners(ScoreIterationListener(1))

    val layers = net.layers
    var totalNumParams = 0
    for (i in layers.indices) {
        val nParams = layers[i].numParams()
        println("Number of parameters in layer $i: $nParams")
        totalNumParams += nParams
    }
    println("Total number of network parameters: " + totalNumParams)

    var miniBatchNumber = 0
    for (i in 0 until numEpochs) {
        while (dataSetIterator.hasNext()) {
            val ds = dataSetIterator.next()
            net.fit(ds)
            if (++miniBatchNumber % generateSamplesEveryNMinibatches == 0) {
                println("--------------------")
                println("Completed " + miniBatchNumber + " minibatches of size " + miniBatchSize + "x" + exampleLength + " characters")
//                println("Sampling characters from network given initialization \"" + (if (generationInitialization == null) "" else generationInitialization) + "\"")
//                val samples = sampleCharactersFromNetwork(generationInitialization, net, dataSetIterator, rng, nCharactersToSample, nSamplesToGenerate)
//                for (j in samples.indices) {
//                    println("----- Sample $j -----")
//                    println(samples[j])
//                    println()
//                }
            }
        }

        dataSetIterator.reset()    //Reset iterator for another epoch
    }

    println("\n\nExample complete")

    while (dataSetIterator.hasNext()) {
        net.fit(dataSetIterator.next())
    }

}