package com.example.tsvad.ui

import android.Manifest
import androidx.compose.animation.animateColorAsState
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.tsvad.MainViewModel

@Composable
fun EnrollScreen(
    viewModel: MainViewModel = viewModel(),
) {
    val state by viewModel.enrollState.collectAsState()

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(32.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
    ) {
        Text(
            text = "Speaker Enrollment",
            style = MaterialTheme.typography.headlineMedium,
        )

        Spacer(modifier = Modifier.height(16.dp))

        Text(
            text = when (state) {
                EnrollState.Idle -> "Record your voice for 5 seconds\nto register your speaker profile."
                EnrollState.Recording -> "Recording... Please speak naturally."
                EnrollState.Processing -> "Extracting speaker embedding..."
                EnrollState.Done -> "Enrollment complete!"
                is EnrollState.Error -> "Error: ${(state as EnrollState.Error).message}"
            },
            style = MaterialTheme.typography.bodyLarge,
            textAlign = TextAlign.Center,
        )

        Spacer(modifier = Modifier.height(48.dp))

        // Recording indicator
        val indicatorColor by animateColorAsState(
            targetValue = when (state) {
                EnrollState.Recording -> Color.Red
                EnrollState.Done -> Color(0xFF4CAF50)
                else -> Color.Gray
            },
            label = "indicator",
        )

        Box(
            modifier = Modifier
                .size(120.dp)
                .clip(CircleShape)
                .background(indicatorColor.copy(alpha = 0.2f)),
            contentAlignment = Alignment.Center,
        ) {
            Box(
                modifier = Modifier
                    .size(80.dp)
                    .clip(CircleShape)
                    .background(indicatorColor.copy(alpha = 0.5f)),
            )
        }

        Spacer(modifier = Modifier.height(48.dp))

        when (state) {
            EnrollState.Idle, is EnrollState.Error -> {
                Button(
                    onClick = { viewModel.startEnrollment() },
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(56.dp),
                ) {
                    Text("Start Recording", fontSize = 18.sp)
                }
            }
            EnrollState.Recording -> {
                CircularProgressIndicator()
            }
            EnrollState.Processing -> {
                CircularProgressIndicator()
            }
            EnrollState.Done -> {
                // Auto-transitions to DetectionScreen via enrollState
                CircularProgressIndicator()
            }
        }
    }
}

sealed class EnrollState {
    data object Idle : EnrollState()
    data object Recording : EnrollState()
    data object Processing : EnrollState()
    data object Done : EnrollState()
    data class Error(val message: String) : EnrollState()
}
