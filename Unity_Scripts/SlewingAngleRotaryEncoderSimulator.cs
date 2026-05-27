using UnityEngine;
using System.IO;

public class SlewingAngleRotaryEncoderSimulator : MonoBehaviour
{
    private float PPR = 1024f; // Pulses Per Revolution
    private float previousAngle = 180f; // Previous slewing angle
    private float currentAngle = 180f; // Current slewing angle
    private float deltaAngle;
    private float accumulatedAngleA;
    private float accumulatedAngleB;

    private float anglePerPulse;

    public crane_animate2 craneScript; // Reference to the crane_animate2 script (set in the Inspector)

    private bool signalA = false;
    private bool signalB = false;
    private bool prevSignalA = false;
    private bool signalAChanged = false;
    private int direction = 0;

    public CSVWriter csvWriter; // Assign this in the Inspector

    private float slewingAngle = 0f; // Variable to store the calculated slewing angle

    void Start()
    {
        // Initialize previous angle with the starting position
        previousAngle = (craneScript.adjustedRotateYaw + 360) % 360;
        anglePerPulse = 360f / PPR;
        accumulatedAngleB = anglePerPulse / 2f;
    }

    void Update()
    {
        // Access the adjustedRotateYaw variable from the crane_animate2 script
        currentAngle = (craneScript.adjustedRotateYaw + 360) % 360;

        // Calculate the change in angle since the last frame
        deltaAngle = Mathf.DeltaAngle(previousAngle, currentAngle);  // Calculate the shortest angular path

        // Accumulate the deltaAngle
        accumulatedAngleA += deltaAngle;
        accumulatedAngleB += deltaAngle;

        // Check if the accumulated angle exceeds the angle per pulse
        // Handling Signal A
        if (Mathf.Abs(accumulatedAngleA) >= anglePerPulse)
        {
            accumulatedAngleA %= anglePerPulse;
            // Toggle the state of Signal A
            signalA = !signalA;
            signalAChanged = true;
        }

        // Handling Signal B
        if (Mathf.Abs(accumulatedAngleB) >= anglePerPulse)
        {
            accumulatedAngleB %= anglePerPulse;
            // Toggle the state of Signal B
            signalB = !signalB;
        }

        // Update slewing angle based on signal changes using quadrature decoding
        if (signalAChanged)
        {
            direction = CalculateDirection(signalA, signalB, prevSignalA);
            slewingAngle += direction * anglePerPulse;

            signalAChanged = false;

            // Update the centralized CSV writer with the latest rotary encoder signals
            csvWriter.UpdateRotaryEncoderSignals(signalA, signalB, slewingAngle);

            // Raise the event
            EncoderSignalChangeNotifier.SignalAChanged();
        }
        
        // Reset previous signals
        prevSignalA = signalA;
        // Update the previous angle
        previousAngle = currentAngle;
    }

    int CalculateDirection(bool signalA, bool signalB, bool prevSignalA)
    {
        // Simplified logic to determine the direction
        if (signalA != prevSignalA) // Signal A has changed
        {
            return (signalA == signalB) ? -1 : 1;
        }
        return 0;
    }
}



/// Slewing Angle rotary encoder data synthesizer calculator before combining the csv outputs into one export

// using UnityEngine;
// using System.IO;

// public class SlewingAngleRotaryEncoderSimulator : MonoBehaviour
// {
//     private float PPR = 1024f; // Pulses Per Revolution
//     private float previousAngle = 180f; // Previous slewing angle
//     private float currentAngle = 180f; // Current slewing angle
//     private float deltaAngle;
//     private float accumulatedAngleA;
//     private float accumulatedAngleB;

//     private float anglePerPulse;

//     public crane_animate2 craneScript; // Reference to the crane_animate2 script (set in the Inspector)

//     private bool signalA = false;
//     private bool signalB = false;
//     private bool prevSignalA = false;
//     private bool signalAChanged = false;
//     private int direction = 0;

//     private string filePath;
//     private float startTime;

//     private float slewingAngle = 0f; // Variable to store the calculated slewing angle

//     void Start()
//     {
//         // Generate timestamp for the file name
//         string timestamp = System.DateTime.Now.ToString("yyyyMMddHHmmss");

//         // Set the file path to the Desktop with timestamp
//         filePath = Path.Combine(System.Environment.GetFolderPath(System.Environment.SpecialFolder.Desktop), $"SlewingAngleRotaryEncoderData_{timestamp}.csv");

//         // Initialize previous angle with the starting position
//         previousAngle = (craneScript.adjustedRotateYaw + 360) % 360;
//         anglePerPulse = 360f / PPR;
//         accumulatedAngleB = anglePerPulse / 2f;

//         // Write the header to the CSV file
//         if (!File.Exists(filePath))
//         {
//             File.WriteAllText(filePath, "TimeFrame,SignalASlewingAngleRotaryEncoder,SignalBSlewingAngleRotaryEncoder,SlewingAngle\n");
//         }

//         // Record the start time
//         startTime = Time.time;
//     }

//     void Update()
//     {
//         // Access the adjustedRotateYaw variable from the crane_animate2 script
//         currentAngle = (craneScript.adjustedRotateYaw + 360) % 360;

//         // Calculate the change in angle since the last frame
//         deltaAngle = Mathf.DeltaAngle(previousAngle, currentAngle);  // Calculate the shortest angular path

//         // Accumulate the deltaAngle
//         accumulatedAngleA += deltaAngle;
//         accumulatedAngleB += deltaAngle;

//         // Check if the accumulated angle exceeds the angle per pulse
//         // Handling Signal A
//         if (Mathf.Abs(accumulatedAngleA) >= anglePerPulse)
//         {
//             accumulatedAngleA %= anglePerPulse;
//             // Toggle the state of Signal A
//             signalA = !signalA;
//             signalAChanged = true;
//         }

//         // Handling Signal B
//         if (Mathf.Abs(accumulatedAngleB) >= anglePerPulse)
//         {
//             accumulatedAngleB %= anglePerPulse;
//             // Toggle the state of Signal B
//             signalB = !signalB;
//         }

//         // Update slewing angle based on signal changes using quadrature decoding
//         if (signalAChanged)
//         {
//             direction = CalculateDirection(signalA, signalB, prevSignalA);
//             slewingAngle += direction * anglePerPulse;

//             signalAChanged = false;

//             // Append the current timeframe, Signal A, Signal B, and Slewing Angle values to the CSV file
//             float currentTimeFrame = Time.time - startTime;
//             string newLine = $"{currentTimeFrame:F2},{(signalA ? 1 : 0)},{(signalB ? 1 : 0)},{slewingAngle}\n";
//             File.AppendAllText(filePath, newLine);
//         }

//         // Update the previous signals
//         prevSignalA = signalA;
//         // Update the previous angle
//         previousAngle = currentAngle;
//     }

//     int CalculateDirection(bool signalA, bool signalB, bool prevSignalA)
//     {
//         // Simplified logic to determine the direction
//         if (signalA != prevSignalA) // Signal A has changed
//         {
//             return (signalA == signalB) ? -1 : 1;
//         }
//         return 0;
//     }
// }
