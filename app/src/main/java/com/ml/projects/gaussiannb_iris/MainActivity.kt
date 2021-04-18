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

// Welcome
// If you've started from the MainActivity, you may view each class in the following order,
// 1. FeatureColumn.kt
// 2. DataFrame.kt
// 3. GaussianNB.kt
class MainActivity : AppCompatActivity() {

    // View elements to take the inputs and display it to the user.
    private lateinit var sampleInputEditText : TextInputEditText
    private lateinit var outputTextView : TextView

    // DataFrame object which will hold the data.
    private lateinit var dataFrame: DataFrame

    // GaussianNB object to perform the calculations and return
    // the predictions.
    private lateinit var gaussianNB: GaussianNB


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView( R.layout.activity_main )

        // Initialize the View elements
        sampleInputEditText = findViewById( R.id.sample_input_editText )
        outputTextView = findViewById( R.id.output_textView )

        // Initialize the DataFrame object.
        // The file iris_ds.csv is kept in the assets folder of the app.
        dataFrame = DataFrame( this , "iris_ds.csv" )

    }

    // Called when predict_button is clicked. ( See activity_main.xml ).
    fun onPredictButtonClick( view : View ) {
        // Split the String by ","
        var strX = sampleInputEditText.text.toString().split( "," ).toTypedArray()
        strX = strX.map{ xi -> xi.trim() }.toTypedArray()
        // Convert the String[] to float[]
        val x = dataFrame.convertStringArrayToFloatArray( strX )
        // Predict the class with GaussianNB.
        gaussianNB.predict( x , resultCallback )
    }

    // Called when load_data_button is clicked. ( See activity_main.xml ).
    fun onLoadCSVButtonClick( view : View ) {
        // Load the CSV file.
        dataFrame.readCSV( readCSVCallback )
    }

    // Called when the CSV data has been processed.
    private val readCSVCallback = object : DataFrame.ReadCSVCallback{

        override fun onCSVProcessed(processedData: Array<FeatureColumn>, labels: Array<String>) {
            // Notify the user that the data has been processed.
            Toast.makeText( this@MainActivity , "CSV File loaded." , Toast.LENGTH_LONG ).show()
            // Initialize the GaussianNB object.
            // We initialize it here because it accesses some variables in dataFrame which are only initialized
            // when this method ( onCSVProcessed ) is called.
            gaussianNB = GaussianNB( dataFrame )
        }

    }

    // Called when all calculations are done and the results could be processed
    // further.
    private val resultCallback = object : GaussianNB.ResultCallback {

        override fun onPredictionResult(label: String, probs: FloatArray) {
            // Display the output to the user via outputTextView.
            outputTextView.text = "Label : $label \n Prob : ${probs.contentToString()}"
        }

    }

}