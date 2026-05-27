using System;
using UnityEngine;

public class EncoderSignalChangeNotifier : MonoBehaviour
{
    public static event Action OnSignalAChanged;

    public static void SignalAChanged()
    {
        OnSignalAChanged?.Invoke();
    }
}
