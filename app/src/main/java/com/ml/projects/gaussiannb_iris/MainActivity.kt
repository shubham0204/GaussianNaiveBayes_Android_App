package com.ml.projects.gaussiannb_iris

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.text.Editable
import android.text.TextWatcher
import android.view.View
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.core.widget.addTextChangedListener
import com.google.android.material.textfield.TextInputEditText
import com.ml.projects.gaussiannb_iris.data.DataFrame
import com.ml.projects.gaussiannb_iris.data.FeatureColumn

class MainActivity : AppCompatActivity() {

    private lateinit var sampleInputEditText : TextInputEditText
    private lateinit var outputTextView : TextView
    private lateinit var predictButton : Button

    private lateinit var dataFrame: DataFrame

    private lateinit var gaussianNB: GaussianNB


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView( R.layout.activity_main )

        sampleInputEditText = findViewById( R.id.sample_input_editText )
        predictButton = findViewById( R.id.predict_button )
        outputTextView = findViewById( R.id.output_textView )

        sampleInputEditText.addTextChangedListener{ object : TextWatcher {

            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {
                TODO("Not yet implemented")
            }

            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {
                predictButton.isEnabled = s?.length != 0
            }

            override fun afterTextChanged(s: Editable?) {
                TODO("Not yet implemented")
            }

        }}

        dataFrame = DataFrame( this , "iris_ds.csv" )

    }

    private val readCSVCallback = object : DataFrame.ReadCSVCallback{
        override fun onCSVProcessed(processedData: Array<FeatureColumn>, labels: Array<String>) {
            Toast.makeText( this@MainActivity , "CSV File loaded." , Toast.LENGTH_LONG ).show()
            gaussianNB = GaussianNB( dataFrame )
        }
    }

    private val resultCallback = object : GaussianNB.ResultCallback {
        override fun onPredictionResult(label: String, probs: FloatArray) {
            outputTextView.text = "Label : $label \n Prob : ${probs.contentToString()}"
        }
    }

    fun onPredictButtonClick( view : View ) {
        var strX = sampleInputEditText.text.toString().split( "," ).toTypedArray()
        strX = strX.map{ xi -> xi.trim() }.toTypedArray()
        val x = dataFrame.convertStringArrayToFloatArray( strX )
        gaussianNB.predict( x , resultCallback )
    }

    fun onLoadCSVButtonClick( view : View ) {
        dataFrame.readCSV( readCSVCallback )
    }

}