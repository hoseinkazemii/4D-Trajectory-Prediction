using System.IO;
using UnityEngine;

public class CSVWriter : MonoBehaviour
{
    private string filePath;
    private StreamWriter writer;
    private bool isWriting = true; // Flag to control writing

    private float currentTimeFrame;
    private float x, y, z, sway;
    private int loadingStarted;
    private int signalASlewingAngle, signalBSlewingAngle;
    private float slewingAngle;
    private int signalARadius, signalBRadius;
    private float radius;
    private int signalAHookHeight, signalBHookHeight;
    private float hookHeight;
    private string header;

    public void InitializeCSVWriter(string scenarioName)
    {
        // Generate timestamp for the file name
        string timestamp = System.DateTime.Now.ToString("yyyyMMddHHmmss");

        // Set the file path with scenario name and timestamp
        filePath = Path.Combine(System.Environment.GetFolderPath(System.Environment.SpecialFolder.Desktop), $"LoadData_{scenarioName}_{timestamp}.csv");

        // Initialize writer
        writer = new StreamWriter(filePath);

        // Write the header immediately after initializing the writer
        header = "Time,X,Y,Z,Sway,LoadingStarted," +
                "SignalASlewingAngleRotaryEncoder,SignalBSlewingAngleRotaryEncoder,SlewingAngle," +
                "SignalARadiusLinearEncoder,SignalBRadiusLinearEncoder,Radius," +
                "SignalAHookHeightLinearEncoder,SignalBHookHeightLinearEncoder,HookHeight";
        WriteHeader(header);
    }

    private void WriteHeader(string header)
    {
        if (writer != null)
        {
            writer.WriteLine(header);
        }
    }

    public void UpdateMetrics(float time, float x, float y, float z, float sway, bool loadingStarted)
    {
        this.currentTimeFrame = time;
        this.x = x;
        this.y = y;
        this.z = z;
        this.sway = sway;
        this.loadingStarted = loadingStarted ? 1 : 0;
    }

    public void UpdateRotaryEncoderSignals(bool signalA, bool signalB, float angle)
    {
        this.signalASlewingAngle = signalA ? 1 : 0;
        this.signalBSlewingAngle = signalB ? 1 : 0;
        this.slewingAngle = angle;

        WriteLine();
    }

    public void UpdateRadiusEncoderSignals(bool signalA, bool signalB, float radius)
    {
        this.signalARadius = signalA ? 1 : 0;
        this.signalBRadius = signalB ? 1 : 0;
        this.radius = radius;

        WriteLine();
    }

    public void UpdateHookHeightEncoderSignals(bool signalA, bool signalB, float height)
    {
        this.signalAHookHeight = signalA ? 1 : 0;
        this.signalBHookHeight = signalB ? 1 : 0;
        this.hookHeight = height;

        WriteLine();
    }

    private void WriteLine()
    {
        if (writer != null && isWriting)
        {
            string newLine = $"{currentTimeFrame:F2},{x:F2},{y:F2},{z:F2},{sway:F2},{loadingStarted},{signalASlewingAngle},{signalBSlewingAngle},{slewingAngle:F2},{signalARadius},{signalBRadius},{radius:F2},{signalAHookHeight},{signalBHookHeight},{hookHeight:F2}";
            writer.WriteLine(newLine);
            writer.Flush();
        }
    }

    // Method to stop writing when the task ends
    public void StopWriting()
    {
        isWriting = false; // Set the flag to stop writing
        if (writer != null)
        {
            writer.Close();
            writer = null; // Release the writer object
        }
    }

    void OnApplicationQuit()
    {
        if (writer != null)
        {
            writer.Close();
        }
    }
}
