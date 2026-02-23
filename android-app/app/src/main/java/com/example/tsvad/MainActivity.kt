package com.example.tsvad

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.tsvad.ui.DetectionScreen
import com.example.tsvad.ui.EnrollScreen
import com.example.tsvad.ui.EnrollState
import com.example.tsvad.ui.theme.TSVADTheme

class MainActivity : ComponentActivity() {

    private val requestPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        hasPermission.value = granted
    }

    private val hasPermission = mutableStateOf(false)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        hasPermission.value = ContextCompat.checkSelfPermission(
            this, Manifest.permission.RECORD_AUDIO
        ) == PackageManager.PERMISSION_GRANTED

        if (!hasPermission.value) {
            requestPermission.launch(Manifest.permission.RECORD_AUDIO)
        }

        setContent {
            TSVADTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background,
                ) {
                    val viewModel: MainViewModel = viewModel()
                    val enrollState by viewModel.enrollState.collectAsState()

                    if (hasPermission.value) {
                        AppContent(viewModel, enrollState)
                    } else {
                        PermissionRequest {
                            requestPermission.launch(Manifest.permission.RECORD_AUDIO)
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun AppContent(viewModel: MainViewModel, enrollState: EnrollState) {
    val showDetection = enrollState == EnrollState.Done

    if (showDetection) {
        DetectionScreen(
            viewModel = viewModel,
            onReEnroll = { viewModel.resetEnrollment() },
        )
    } else {
        EnrollScreen(viewModel = viewModel)
    }
}

@Composable
private fun PermissionRequest(onRequest: () -> Unit) {
    Column(
        modifier = Modifier.fillMaxSize(),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
    ) {
        Text(
            text = "Microphone permission is required for TS-VAD.",
            style = MaterialTheme.typography.bodyLarge,
        )
        Spacer(modifier = Modifier.height(16.dp))
        Button(onClick = onRequest) {
            Text("Grant Permission")
        }
    }
}
