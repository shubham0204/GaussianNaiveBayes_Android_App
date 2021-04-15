package com.ml.projects.gaussiannb_iris.data

import kotlin.math.pow

class FeatureColumn( var name : String , var data : FloatArray ) {

    var featureMean : Float
    var featureVariance : Float
    var featureStdDev : Float

    init {
        featureMean = computeMean()
        featureVariance = computeVariance()
        featureStdDev = featureVariance.pow( 0.5f )
    }

    override fun toString(): String {
        return "{ name = $name , mean = $featureMean , stddev = $featureStdDev }"
    }

    private fun computeMean() : Float {
        return data.average().toFloat()
    }

    private fun computeVariance() : Float {
        val mean = data.average().toFloat()
        val meanSquaredSum = data.map{ xi -> ( xi - mean ).pow( 2 ) }.sum()
        return meanSquaredSum / data.size
    }

}