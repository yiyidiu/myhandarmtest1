#include <cmath>

#include <QtCore/QTimer>

#include <gazebo/gui/GuiEvents.hh>
#include <gazebo/gui/GuiPlugin.hh>

namespace gazebo
{
class GuiRenderRateLimiterPlugin : public GUIPlugin
{
public:
  GuiRenderRateLimiterPlugin()
  {
    // This plugin has no user interface.  Gazebo unconditionally shows GUI
    // plugins after Load(), so make the intermediate widget harmless and hide
    // it again on the next Qt event-loop turn.
    this->setFixedSize(1, 1);
    this->move(-1, -1);
    this->setAttribute(Qt::WA_TransparentForMouseEvents, true);
    this->setStyleSheet("background: transparent;");
  }

  void Load(sdf::ElementPtr sdf) override
  {
    double render_rate_hz = 15.0;
    if (sdf && sdf->HasElement("renderRateHz"))
      render_rate_hz = sdf->GetElement("renderRateHz")->Get<double>();

    if (!std::isfinite(render_rate_hz) || render_rate_hz < 1.0 ||
        render_rate_hz > 120.0)
    {
      gzerr << "GuiRenderRateLimiterPlugin renderRateHz must be in "
            << "[1, 120]; refusing invalid value " << render_rate_hz << "\n";
      QTimer::singleShot(0, this, [this]() { this->hide(); });
      return;
    }

    // GLWidget subscribes to this in its constructor.  World-provided GUI
    // plugins are loaded after the scene and user camera exist, so the event
    // updates both the UserCamera render period and GLWidget's update timer.
    gui::Events::setRenderRate(render_rate_hz);
    gzmsg << "Teleoperation Gazebo GUI render rate limited to "
          << render_rate_hz << " Hz\n";
    QTimer::singleShot(0, this, [this]() { this->hide(); });
  }
};

GZ_REGISTER_GUI_PLUGIN(GuiRenderRateLimiterPlugin)
}  // namespace gazebo
